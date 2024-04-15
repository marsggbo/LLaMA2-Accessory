import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])

from typing import List, Optional, Tuple, Union
from accessory.model.meta import MetaModel

import time
import argparse
import torch
import torch.distributed as dist
from torch.autograd import profiler as torch_profiler
import numpy as np
import random
import json

from accessory.util import misc
from fairscale.nn.model_parallel import initialize as fs_init

from accessory.data.alpaca import format_prompt
from accessory.util.tensor_parallel import load_tensor_parallel_model_list
from accessory.util.tensor_type import default_tensor_type

from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)
from dataclasses import dataclass
from examples.scheduler import PatternScheduler

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
        return batch


def prepare_dataset(
        data_list: Optional[int] = 0,
        data_size: int = 2000,
        sort_by_len: bool = True,
    ):
    '''
    Args:
        data_list:
            list[string]
            0: alpaca data list
            1: yizhongw data list
            2: alpaca-yizhongw combined data list
    '''
    def load_json(file):
        with open(file, 'r') as f:
            data = json.load(f)
        return data

    if data_list == 0:
        alpaca_data = load_json("/home/nus-hx/code/Sequence-Scheduling/data/alpaca-train-10k.json")
        data_list = []
        for i in range(data_size):
            data_list.append(alpaca_data[i]['conversations'][0]['value'])
    elif data_list == 1:
        yizhongw_data = load_dataset("yizhongw/self_instruct", "super_natural_instructions")
        data_prompts = yizhongw_data['train']['prompt']
        data_list = []
        for i in range(data_size):
            data_list.append(data_prompts[i])
    elif data_list == 2:
        data_list = []
        alpaca_data = load_json("/home/nus-hx/code/Sequence-Scheduling/data/alpaca-train-10k.json")
        for i in range(data_size):
            data_list.append(alpaca_data[i]['conversations'][0]['value'])

        yizhongw_data = load_dataset("yizhongw/self_instruct", "super_natural_instructions")
        data_prompts = yizhongw_data['train']['prompt']
        for i in range(data_size):
            data_list.append(data_prompts[i])

    if sort_by_len:
        data_list = sorted(data_list, key=len)
    data = {"sentence": data_list}
    dataset = Dataset.from_dict(data)
    return dataset


def get_args_parser():
    parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
    # Model parameters
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                        help='directory containing pretrained checkpoints')
    parser.add_argument('--llama_type', default=None, type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default=None, type=str, nargs="*",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='path to tokenizer.model')

    parser.add_argument('--max_seq_len', type=int, default=4096)

    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
                        help="The dtype used for model weights and inference.")
    parser.add_argument('--quant', action='store_true', help="enable quantization")

    parser.add_argument('--dist_on_itp', action='store_true')
    return parser


@ torch.inference_mode()
def generate(
    model,
    prompt,
    question_input,
    system_prompt,
    max_gen_len,
    gen_t, top_p
):
    image = None

    # text output
    _prompt = format_prompt({"instruction":prompt, "input":question_input}, system_prompt)

    dist.barrier()
    dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p])
    if args.quant:
        results = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
    else:
        with torch.cuda.amp.autocast(dtype=target_dtype):
            results = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
    text_output = results[0].strip()
    return text_output


def init_env(args):
    # define the model
    random.seed(0)
    torch.random.manual_seed(0)
    np.random.seed(0)
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(dist.get_world_size())

def main():
    args = get_args_parser().parse_args()
    if os.environ.get('ipdb', 0):
        from ipdb import set_trace
        set_trace()
    init_env(args)
    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    print('Loading MoE model and tokenizer')
    model = MetaModel.from_pretrained(args.pretrained_path, args.llama_type, args.llama_config, args.tokenizer_path,
                                    with_visual=False, max_seq_len=args.max_seq_len,
                                    mp_group=fs_init.get_model_parallel_group(),
                                    dtype=target_dtype, device="cpu" if args.quant else "cuda",)

    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # with default_tensor_type(dtype=target_dtype, device="cpu" if args.quant else "cuda"):
    #     model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=False)
    # print(f"load pretrained from {args.pretrained_path}")
    # load_result = load_tensor_parallel_model_list(model, args.pretrained_path)
    # print("load result: ", load_result)

    if args.quant:
        print("Quantizing model to 4bit!")
        from accessory.util.quant import quantize
        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig.from_dict(
            config_dict={
                "load_in_8bit": False, 
                "load_in_4bit": True, 
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16
            },
            return_unused_kwargs=False,
        )
        quantize(model, quantization_config)

    # model.llma.layers = model.llma.layers[:8]
    print("Model = %s" % str(model))
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")
    model = model.bfloat16().to(device)
    
    print('Warm up model inference')
    # x = torch.randint(0, 10000, (2, 128)).to(device)
    # out = model.llma.forward_inference(x, 0)
    # print(out.shape)
    # out, _ = model.llma(x)
    # print(out.shape)

    ########################
    # inference text
    ########################
    # preprocess_function = lambda examples: tokenizer(
    #         examples["sentence"], padding=True, return_tensors="pt", return_attention_mask=True)
    # dataset = prepare_dataset(data_list=2, sort_by_len=True)
    # tokenized_dataset = dataset.map(preprocess_function, batched=True)
    # data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)
    # data_loader = DataLoader(
    #     tokenized_dataset,
    #     batch_size=64,
    #     shuffle=True,
    #     collate_fn=data_collator
    # )
    # for batch_id, batch_data in enumerate(data_loader):
    #     if batch_id==20:
    #         break
    #     input_ids = batch_data['input_ids'].to(device)
    #     attention_mask = batch_data['attention_mask'].to(device)
    #     out = model.llma.forward_inference(input_ids, 0)
    #     print(input_ids.shape, out.shape, attention_mask.sum(-1).max(), attention_mask.sum(-1).view(-1)[:20])

    
    # dataset = Dataset.load_from_disk('/home/nus-hx/code/vllm/examples/data/sst2_10000_mrpc_2000_MixtralMoE_patterns')
    dataset = load_dataset("marsggbo/sst2_10000_mrpc_2000_MixtralMoE_patterns")['train']
    # dataset = load_dataset("marsggbo/alpaca10k_yizhongw10k_MixtralMoE_patterns")['train']
    dataset.shuffle(seed=1234)

    batch_size = 256 
    num_samples = len(dataset)
    num_samples = batch_size * 10

    # ########################
    # # inference token_ids
    # ########################
    print("Batch size {0}, Total samples: {1}".format(batch_size, num_samples))
    print('Benchmark inference speed of normal order')
    # normal_indices = list(range(num_samples))
    # normal_time_cost = []
    # normal_tokens = []
    # for i, indices in enumerate(np.array_split(normal_indices, num_samples/batch_size)):
    #     # print(indices)
    #     samples = dataset.select(indices)
    #     input_ids = [sample['token_idx'][:sample['prompt_len']] for sample in samples]
    #     input_ids = tokenizer.pad({'input_ids': input_ids}, return_tensors='pt')['input_ids'].to(device)
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     out = model.llma.forward_inference(input_ids, 0)
    #     torch.cuda.synchronize()
    #     end = time.time()
    #     batch_time = end - start
    #     if i > 10:
    #         normal_time_cost.append(batch_time)
    #         normal_tokens.append(input_ids.numel())
    #     if i<20:
    #         print(f"Batch-{i} {input_ids.shape} takes {batch_time:.4f}s")
    # normal_throughput = sum(normal_tokens) / sum(normal_time_cost)
    # print(f'Normal averagely takes {np.mean(normal_time_cost):.4f}s per batch, throughput is {normal_throughput:.4f} token/s')

    normal_indices = list(range(num_samples))
    normal_time_cost = []
    normal_tokens = []
    input_ids_list = []

    # *********** Warmup peroid ***********
    for i, indices in enumerate(np.array_split(normal_indices, num_samples/batch_size)):
        start = time.time()
        samples = dataset.select(indices)
        end = time.time()
        print("Data selection time ", end - start)
        start = time.time()
        input_ids = [sample['token_idx'][:sample['prompt_len']] for sample in samples]
        input_ids = tokenizer.pad({'input_ids': input_ids}, return_tensors='pt')['input_ids']
        end = time.time()
        print("Padding time ", end - start)
        start = time.time()
        input_ids = input_ids.to(device, non_blocking=True)
        input_ids_list.append(input_ids)
        end = time.time()
        print("Input id Copy to GPU time ", end - start)
        out = model.llma.forward_inference(input_ids, 0)
        normal_tokens.append(input_ids.numel())

    # *********** Profile peroid ***********
     # *********** Profile peroid ***********
    total_iter = 1
    torch.cuda.synchronize()
    torch.distributed.barrier()
    start = time.time()
    for iter in range(total_iter):
        # for i, indices in enumerate(np.array_split(normal_indices, num_samples/batch_size)):
        #     samples = dataset.select(indices)
        #     input_ids = [sample['token_idx'][:sample['prompt_len']] for sample in samples]
        #     input_ids = tokenizer.pad({'input_ids': input_ids}, return_tensors='pt')['input_ids'].to(device, non_blocking=True)
        #     out = model.llma.forward_inference(input_ids, 0)
        torch.cuda.cudart().cudaProfilerStart()
        with torch_profiler.emit_nvtx(record_shapes=True):
            for batch_id, input_ids in enumerate(input_ids_list):
                torch.cuda.nvtx.range_push(f'Batch {batch_id}')
                # torch.cuda.synchronize()
                # start = time.time()
                out = model.llma.forward_inference(input_ids, 0)
                torch.cuda.nvtx.range_pop()
                # torch.cuda.synchronize()
                # end = time.time()
                # print(f'Batch {batch_id} time {end - start}')
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    throughput = sum(normal_tokens) * total_iter / (end - start)
    print(f'Throughput is {throughput:.4f} token/s')


    # print('Benchmark inference speed of scheduled order')
    # print('Loading Scheduler...')
    # scheduler = PatternScheduler(
    #     predictor=0, tokenizer=0,
    #     queue_max_length=batch_size,
    #     window_size=10,
    #     gpu_memory_limit=-1,
    #     device=device
    # )
    # matrix_list = []
    # token_idx_list = []
    # for i in range(num_samples):
    #     sample = dataset[i]
    #     prompt_len = sample['prompt_len']
    #     token_idx_list.append(sample['token_idx'][:prompt_len])
    #     matrix_list.append(sample['token_expert_patterns'][:prompt_len])
    # batch_indices_list = []
    # reorder_time_cost = []
    # schedule_time_cost = []
    # step = 0
    # while matrix_list:
    #     start = time.time()
    #     sorted_matrix_indices = scheduler.schedule_pattern_matrix_list(matrix_list)
    #     end = time.time()
    #     reorder_time_cost.append(end - start)
    #     if step<8:
    #         print(f"Scheduling takes {end - start:.4f}s")
    #     batch_indices = sorted_matrix_indices[:batch_size]
    #     input_ids = [token_idx_list[i] for i in batch_indices]
    #     input_ids = tokenizer.pad({'input_ids': input_ids}, return_tensors='pt')['input_ids'].to(device)
    #     batch_indices_list.append(batch_indices)
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     out = model.llma.forward_inference(input_ids, 0)
    #     torch.cuda.synchronize()
    #     end = time.time()
    #     batch_time = end - start
    #     schedule_time_cost.append(batch_time)
    #     if step<8:
    #         print(f"Batch-{step} takes {batch_time:.4f}s")
    #     step += 1
    #     matrix_list = [matrix_list[i] for i in range(len(matrix_list)) if i not in batch_indices]
    #     token_idx_list = [token_idx_list[i] for i in range(len(token_idx_list)) if i not in batch_indices]
    # schedule_throughput = num_samples / np.sum(schedule_time_cost)
    # print(f"Reorder averagely takes {np.mean(reorder_time_cost):.4f}s")
    # print(f"Scheduler averagely takes {np.mean(schedule_time_cost):.4f}, throughput is {schedule_throughput:.4f} tokens/s.")



if __name__ == '__main__':
    main()

# 8 GPUs
# torchrun --nproc-per-node=8 --master-port=6869 test_mixtral.py --pretrained_path /data/personal/nus-hx/huggingface/hub/MoE-Mixtral-7B-8Expert 
# DEBUG=1 CUDA_VISIBLE_DEVICES=5 torchrun --nproc-per-node=1 --master-port=6869 test_mixtral.py --pretrained_path /data/personal/nus-hx/huggingface/hub/MoE-Mixtral-7B-8Expert 

# EXPERT_DISTRIBUTION="uniform" torchrun --nproc-per-node=8 --master-port=6869 test_mixtral.py --pretrained_path /data/personal/nus-hx/huggingface/hub/MoE-Mixtral-7B-8Expert 
