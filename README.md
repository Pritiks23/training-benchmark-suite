# Training Efficiency Benchmark Suite

A systems benchmarking framework comparing:

- PyTorch FSDP
- Hugging Face + DeepSpeed
- NVIDIA NeMo (Megatron-style training)

## Metrics

- GPU utilization
- step time (ms)
- memory usage / fragmentation
- throughput (tokens/sec)
- convergence (loss curves)

## Design

All frameworks share:
- same model
- same dataset
- same optimizer setup (as close as possible)

## Run

### FSDP
bash scripts/run_fsdp.sh

### DeepSpeed
bash scripts/run_deepspeed.sh

### NeMo
bash scripts/run_nemo.sh

## Output

All logs saved as JSONL in /results
