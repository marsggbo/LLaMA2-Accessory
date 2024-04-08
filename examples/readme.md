1. 下载模型权重

```bash
bash download_weights.sh
```

2. 下载测试数据集

- https://huggingface.co/datasets/marsggbo/sst2_10000_mrpc_2000_MixtralMoE_patterns
- https://huggingface.co/datasets/marsggbo/alpaca10k_yizhongw10k_MixtralMoE_patterns

```python
from datasets import load_dataset

dataset = load_dataset("marsggbo/alpaca10k_yizhongw10k_MixtralMoE_patterns")
```

3. 运行

```python
torchrun --nproc-per-node=8 --master-port=6869 test_mixtral.py --pretrained_path /data/personal/nus-hx/huggingface/hub/MoE-Mixtral-7B-8Expert 
```