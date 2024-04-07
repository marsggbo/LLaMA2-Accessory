import time
from collections import deque
from typing import List, Optional, Tuple, Union, Deque
import types

import numpy as np
import torch
import torch.nn as nn

# from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        if not hasattr(self, 'padding_side'):
            self.padding_side = 'right'
        assert self.padding_side in ['left', 'right'], "Padding should be on one side (left or right)"
        non_label_features =[]
        for feature in features:
            item = {key: val for key, val in feature.items() if key in ['input_ids', 'attention_mask']}
            non_label_features.append(item)
        batch = super().__call__(non_label_features)
        batch['sequence_group'] = [feature['sequence_group'] for feature in features]
        return batch


class CustomDataset(Dataset):
    def __init__(self, sequence_group_list):
        self.sequence_group_list = sequence_group_list
        data_list = []
        for seq_group in sequence_group_list:
            # 获取当前序列组的序列长度
            current_length = seq_group.get_seqs()[0].get_len()

            # 如果 seq_group 没有 sentence_pattern 属性，说明它处于 prefilling 阶段且未被预测过
            if not hasattr(seq_group, "sentence_pattern"):
                seq_group.num_tokens_for_sentence_pattern = current_length
                data_list.append(seq_group)
            # 对于已有 sentence_pattern 的 seq_group，需要判断是否处于 decoding 阶段且有新 token 产生
            elif seq_group.num_tokens_for_sentence_pattern != current_length:
                # 更新 num_tokens_for_sentence_pattern 以反映新的 token 数量
                seq_group.num_tokens_for_sentence_pattern = current_length
                data_list.append(seq_group)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 获取单个数据对象
        seq_group = self.data_list[idx]
        input_ids = torch.tensor(seq_group.get_seqs()[0].get_token_ids()).int()
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sequence_group': seq_group
        }


def new_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    num_experts_per_layer = 8,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = []
        for i in range(len(self.lm_head)):
            logits_per_expert = self.lm_head[i](hidden_states).float() # (bs, seq_len, num_experts)
            logits.append(logits_per_expert)
        logits = torch.stack(logits, dim=-2) # # (bs, seq_len, num_layers, num_experts)

        loss = None
        if labels is not None:
            # masked BCE loss
            labels = labels.to(logits.device).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.view(-1, num_experts_per_layer),
                labels.view(-1, num_experts_per_layer),
                reduction='none')
            loss_mask = labels.view(-1, num_experts_per_layer).sum(-1) != 0
            loss = loss[loss_mask].sum() / loss_mask.sum()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PatternScheduler:
    def __init__(
        self,
        predictor: torch.nn.Module=None,
        tokenizer: AutoTokenizer=None,
        batch_size_for_predictor: int = 128,
        start_seq_index: int = 0,
        window_size: int=300,
        queue_max_length: int=100,
        gpu_memory_limit: int=64*256,
        length_sort: bool = True,
        device='cpu'
    ):
        self.device = device
        if predictor is not None:
            self.predictor = predictor
        if tokenizer is not None:
            self.tokenizer = tokenizer
        if predictor is None or tokenizer is None:
            self.predictor, self.tokenizer = self.prepare_model_tokenizer()
        self.data_collator = CustomDataCollatorWithPadding(tokenizer=self.tokenizer)
        self.batch_size_for_predictor = batch_size_for_predictor
        self.start_seq_index = start_seq_index
        self.window_size = window_size
        if queue_max_length == -1:
            queue_max_length = np.inf # 如果设置为-1，则表示队列长度不限制，直到把 GPU 内存用满为止
        self.queue_max_length = queue_max_length
        if gpu_memory_limit == -1:
            gpu_memory_limit = np.inf # 如果设置为-1，则表示 GPU 内存不限制，将所有数据视为一个 batch 进行调度排序
        self.gpu_memory_limit = gpu_memory_limit
        self.length_sort = length_sort

    def prepare_model_tokenizer(self):
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )
        from safetensors.torch import load_file
        
        num_layers = 32
        num_experts_per_layer = 8
        NUM_LABELS = num_layers * num_experts_per_layer
        PADDING_SIDE = 'right'
        model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
        ckpt_path = "/home/nus-hx/code/vllm/examples/ckpts/finetuneAll_AlltrainMax512StartRandom_EvalMax512_2Layer_bceIgnore_baselr2e-5wd1e-4_head1e-3wd5e-3/checkpoint-5625/model.safetensors"
        print('building predictor...')
        predictor_num_layers = 2
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            cache_dir="/data/common/mixtral/")
        model.model.layers = model.model.layers[:predictor_num_layers]
        model.forward = types.MethodType(new_forward, model)
        model.lm_head = nn.ModuleList([
            nn.Linear(model.config.hidden_size, 8, bias=False) for i in range(32)
        ])
        print('loading weights for prediction...')
        loaded_weights = load_file(ckpt_path)
        model.load_state_dict(loaded_weights)

        print('building tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side=PADDING_SIDE
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        model = model.to(self.device)
        model.eval()
        return model, tokenizer

    def row_normalize(self, x):
        # row_min = np.min(x, axis=-1, keepdims=True)
        # row_max = np.max(x, axis=-1, keepdims=True)
        # return (x - row_min) / (row_max - row_min + 1e-8)
        return x

    def calculate_combined_variance(self, candidate_patterns, combined_pattern):
        # 计算合并后的pattern矩阵的方差
        normalized_combined = self.row_normalize(combined_pattern.unsqueeze(0) + candidate_patterns)
        variances = torch.mean(torch.var(normalized_combined, dim=-1), dim=-1)
        return variances

    def convert_logits_to_pattern(self, outputs, attention_mask):
        '''Convert predictor outputs to pattern
        '''
        logits = outputs.logits
        top2_indices = logits.topk(2, dim=-1).indices
        top2_one_hot_pattern_predictions = torch.zeros_like(logits)
        top2_one_hot_pattern_predictions.scatter_(-1, top2_indices, 1)
        extended_attention_mask = attention_mask[:, :, None, None]
        masked_output = top2_one_hot_pattern_predictions * extended_attention_mask
        sequence_level_patterns = torch.sum(masked_output, dim=1)
        return sequence_level_patterns

    def prepare_dataloader(self, dataset, batch_size=16, collate_fn=None):
        if collate_fn is None:
            collate_fn = self.data_collator
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        return dataloader

    def schedule(self, sequence_group_list, length_sort=None, device='cuda:0'):
        dataset = CustomDataset(sequence_group_list)
        batch_size = min(self.batch_size_for_predictor, len(dataset))
        if batch_size > 0:
            dataloader = self.prepare_dataloader(dataset, batch_size=batch_size, collate_fn=self.data_collator)
            #  使用 self.predictor 预测每个 sequence_group 的 sentence_pattern
            with torch.inference_mode():
                for batch_id, batch_data in enumerate(dataloader):
                    batch_inputs = {k: v.to(device) for k, v in batch_data.items() if k != 'sequence_group'}
                    batch_size = batch_inputs['input_ids'].size(0)
                    outputs = self.predictor(**batch_inputs)
                    sequence_level_patterns = self.convert_logits_to_pattern(outputs, batch_inputs['attention_mask'])
                    for i in range(batch_size):
                        batch_data['sequence_group'][i].sentence_pattern = sequence_level_patterns[i]
        candidate_patterns = [seq_group.sentence_pattern for seq_group in sequence_group_list]
        self.candidate_patterns = torch.stack(candidate_patterns)

        start_index = self.start_seq_index
        combined_pattern = self.candidate_patterns[start_index].clone()
        scheduled_indices = [start_index]
        searched_seq_indices = [start_index]
        total_tokens = sequence_group_list[start_index].get_seqs()[0].get_len()
        if length_sort is None:
            length_sort = self.length_sort
        if length_sort:
            # 根据句子长度从短到长排序
            seq_length_list = [seq_group.get_seqs()[0].get_len() for seq_group in sequence_group_list]
            indices_to_search = np.argsort(seq_length_list)
        else:
            indices_to_search = np.arange(len(sequence_group_list))

        while len(scheduled_indices) < self.queue_max_length and len(searched_seq_indices) < len(sequence_group_list): # 只要完成一个 batch 的数据排序即可，效率更高
            # 限制搜索范围到滑动窗口内的句子
            mask = np.isin(indices_to_search, searched_seq_indices, invert=True) # 保持原本的相关顺序
            window_indices = indices_to_search[mask][:self.window_size]
            candidate_patterns = self.candidate_patterns[window_indices]

            # 计算方差
            variances = self.calculate_combined_variance(candidate_patterns, combined_pattern)

            # 选择方差最小的句子索引
            min_variance_index = window_indices[torch.argmin(variances).item()]

            inner_flag1 = len(scheduled_indices) < self.queue_max_length # 没有超出队列长度限制
            inner_flag2 = total_tokens+sequence_group_list[min_variance_index].get_seqs()[0].get_len() < self.gpu_memory_limit # 没有超出 GPU 内存限制
            if inner_flag1 and inner_flag2:
                scheduled_indices.append(min_variance_index)
                combined_pattern += self.candidate_patterns[min_variance_index]
                total_tokens += sequence_group_list[min_variance_index].get_seqs()[0].get_len()
                searched_seq_indices.append(min_variance_index)

        un_scheduled_indices = [i for i in indices_to_search if i not in searched_seq_indices]
        final_indices = scheduled_indices+un_scheduled_indices
        assert len(final_indices) == len(sequence_group_list)
        return deque([sequence_group_list[i] for i in final_indices])

    def calculate_distance_to_uniform(self, candidate_patterns, combined_pattern):
        # 假定 combined_pattern 已经被加到 candidate_patterns 上
        # 首先，计算合并后的 pattern 矩阵的行归一化版本
        normalized_combined = combined_pattern.unsqueeze(0) + candidate_patterns
        normalized_combined = normalized_combined / normalized_combined.sum(-1, keepdim=True)
        
        # 然后，创建一个目标矩阵，其所有元素值均为 0.125
        target_uniform = torch.full_like(normalized_combined, 0.125)
        
        # 计算和目标均匀分布矩阵之间的欧几里得距离
        distance_to_uniform = torch.norm(normalized_combined - target_uniform, dim=-1)
        
        # 计算距离的平均值作为最终的度量
        average_distance = torch.mean(distance_to_uniform, dim=-1)
        
        return average_distance

    def schedule_pattern_matrix_list(self, pattern_matrix_list: List[List], length_sort=None, device='cuda:0'):
        candidate_patterns = []
        for seq_index, seq_token_pattern_list in enumerate(pattern_matrix_list):
            seq_pattern = np.sum(seq_token_pattern_list, axis=0)
            candidate_patterns.append(seq_pattern)
        self.candidate_patterns = torch.from_numpy(np.stack(candidate_patterns)).float().to(device)

        start_index = self.start_seq_index
        combined_pattern = self.candidate_patterns[start_index].clone()
        scheduled_indices = [start_index]
        searched_seq_indices = [start_index]
        total_tokens = len(pattern_matrix_list[start_index])
        if length_sort is None:
            length_sort = self.length_sort
        if length_sort:
            # 根据句子长度从短到长排序
            seq_length_list = [len(seq_token_pattern_list) for seq_token_pattern_list in pattern_matrix_list]
            indices_to_search = np.argsort(seq_length_list)
        else:
            indices_to_search = np.arange(len(pattern_matrix_list))

        while len(scheduled_indices) < self.queue_max_length and len(searched_seq_indices) < len(pattern_matrix_list): # 只要完成一个 batch 的数据排序即可，效率更高
            # 限制搜索范围到滑动窗口内的句子
            mask = np.isin(indices_to_search, searched_seq_indices, invert=True) # 保持原本的相关顺序
            window_indices = indices_to_search[mask][:self.window_size]
            candidate_patterns = self.candidate_patterns[window_indices]

            # 计算方差
            variances = self.calculate_distance_to_uniform(candidate_patterns, combined_pattern)

            # 选择方差最小的句子索引
            min_variance_index = window_indices[torch.argmin(variances).item()]

            inner_flag1 = len(scheduled_indices) < self.queue_max_length # 没有超出队列长度限制
            inner_flag2 = total_tokens+len(pattern_matrix_list[min_variance_index]) < self.gpu_memory_limit # 没有超出 GPU 内存限制
            if inner_flag1 and inner_flag2:
                scheduled_indices.append(min_variance_index)
                combined_pattern += self.candidate_patterns[min_variance_index]
                total_tokens += len(pattern_matrix_list[min_variance_index])
                searched_seq_indices.append(min_variance_index)

        un_scheduled_indices = [i for i in indices_to_search if i not in searched_seq_indices]
        final_indices = scheduled_indices+un_scheduled_indices
        assert len(final_indices) == len(pattern_matrix_list)
        return final_indices

if __name__ == '__main__':
    queue_max_length = 64
    scheduler = PatternScheduler(
        predictor=0, tokenizer=0,
        queue_max_length=queue_max_length,
        gpu_memory_limit=-1,
        device=torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
    )
    from datasets import Dataset, concatenate_datasets

    loaded_dataset1 = Dataset.load_from_disk('/home/nus-hx/code/vllm/examples/data/alpaca10k_yizhongw10k_MixtralMoE_patterns')
    # loaded_dataset2 = Dataset.load_from_disk('/home/nus-hx/code/vllm/examples/data/sst2_10000_mrpc_2000_MixtralMoE_patterns')
    # dataset = concatenate_datasets([loaded_dataset1, loaded_dataset2])
    dataset = loaded_dataset1
    alpaca_dataset = dataset.select(list(range(10000))).shuffle(666)
    yizhongw_dataset = dataset.select(list(range(10000, 20000))).shuffle(666)
    # sst2_dataset = dataset.select(list(range(20000, 30000))).shuffle(666)
    # mrpc_dataset = dataset.select(list(range(30000, len(dataset)))).shuffle(666)

    matrix_list = []
    for i in range(256):
        prompt_len = alpaca_dataset[i]['prompt_len']
        matrix_list.append(alpaca_dataset[i]['token_expert_patterns'][:prompt_len])
    assert len(matrix_list)==256
    sorted_matrix_indices = scheduler.schedule_pattern_matrix_list(matrix_list)
    print(sorted_matrix_indices[:queue_max_length])
