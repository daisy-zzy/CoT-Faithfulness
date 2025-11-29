# 10701-project

### 1. Environment
```
cd 10701-project
pip install -r requirements.txt
huggingface-cli login # log-in to have access to Llama
```

### 2. Running Baselines
```
python run_stage1_qwen.py
python run_stage2_llama.py
```
The results will be saved to `./outputs` automatically.