## DP Fine-tuning

### Setup

We leverage [dp-transformers](https://github.com/microsoft/dp-transformers) to perform DP fine-tuning. The original repository is too old to run smoothly. We have patched the dp-transformers to make it compatiable with the latest Hugging Face softwares.

```bash
cd dp-transformers
pip install -e .
```

### Fine-tune

```bash
accelerate config # It only support vanilla data parallel now. DeepSpeed or FSDP is not usable now.
chmod a+x run-fime-tune-dp-llama.sh
./run-fine-tune-dp-llama.sh
```