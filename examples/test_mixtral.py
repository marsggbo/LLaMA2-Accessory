import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])

from accessory.model.meta import MetaModel

import argparse
import torch
import torch.distributed as dist
import gradio as gr
import numpy as np
import random

from accessory.util import misc
from fairscale.nn.model_parallel import initialize as fs_init

from accessory.data.alpaca import format_prompt
from accessory.util.tensor_parallel import load_tensor_parallel_model_list
from accessory.util.tensor_type import default_tensor_type


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

args = get_args_parser().parse_args()
if os.environ.get('DEBUG', 0):
    from ipdb import set_trace
    set_trace()

# define the model
random.seed(0)
torch.random.manual_seed(0)
np.random.seed(0)
misc.init_distributed_mode(args)
fs_init.initialize_model_parallel(dist.get_world_size())
target_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}[args.dtype]
model = MetaModel.from_pretrained(args.pretrained_path, args.llama_type, args.llama_config, args.tokenizer_path,
                                  with_visual=False, max_seq_len=args.max_seq_len,
                                  mp_group=fs_init.get_model_parallel_group(),
                                  dtype=target_dtype, device="cpu" if args.quant else "cuda",)

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

print("Model = %s" % str(model))
model.bfloat16().cuda()
x = torch.randint(0, 10000, (32, 128)).cuda()
out = model.llma.forward_inference(x, 0)
print(out.shape)
out, _ = model.llma(x)
print(out.shape)

@ torch.inference_mode()
def generate(
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


# DEBUG=1 CUDA_VISIBLE_DEVICES=5 torchrun --nproc-per-node=1 --master-port=6869 test_mixtral.py --pretrained_path /data/personal/nus-hx/huggingface/hub/MoE-Mixtral-7B-8Expert --max_seq_len 1