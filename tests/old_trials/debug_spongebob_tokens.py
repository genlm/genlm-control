import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    prompt = "SpongeBob SquarePants lives in a "
    print(f"Prompt tokens: {tokenizer.encode(prompt)}")
    
    for word in ["pineapple", "house"]:
        full = prompt + word
        print(f"Full tokens ({word}): {tokenizer.encode(full)}")

if __name__ == "__main__":
    main()

