from utils.pipeline import full_pipeline

if __name__ == "__main__":
    full_pipeline(
        qwen_name="Qwen/Qwen3-4B",
        llama_name="meta-llama/Llama-3.2-3B-Instruct",
        out_dir="outputs",
        max_examples=8,  
    )
