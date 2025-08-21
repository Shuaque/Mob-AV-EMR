from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def args():
    parser = argparse.ArgumentParser(description="LLM Inference Script")
    
    parser.add_argument('-i', '--input', type=str, default='None', help='input text')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args()

    model_id = "/workspace/shuaque/Mob-AV-EMR/pretrained/LLM/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [
    {"role":"system","content":"You are Qwen, a helpful assistant."},
    {"role":"user","content": args.input}
    ]

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=256)

    gen = [o[len(i):] for i,o in zip(inputs.input_ids, out_ids)]
    print(tok.batch_decode(gen, skip_special_tokens=True)[0])