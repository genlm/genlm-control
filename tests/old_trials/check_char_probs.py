import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    prompt = "Pick a fruit: ["
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    chars_to_check = ["a", "b", "c", "d", "e", "f", "g", "p"]
    
    print(f"{'Char':<10} | {'TokenID':<10} | {'LogProb':<10}")
    print("-" * 40)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

        for char in chars_to_check:
            # Encode char without space
            # But be careful: tokenizer.encode("a") might be different from what we want?
            # We want the token that IS "a".
            # "a" is 64.
            # "b" is 65.
            # "c" is 66.
            token_id = tokenizer.encode(char)[0] 
            # Verify it decodes to char
            decoded = tokenizer.decode([token_id])
            
            p = probs[token_id].item()
            log_p = torch.log(torch.tensor(p)).item()
            
            print(f"{decoded:<10} | {token_id:<10} | {log_p:.4f}")

        # Also check "apple" token
        apple_id = tokenizer.encode("apple")[0]
        p_apple = probs[apple_id].item()
        print(f"{'apple':<10} | {apple_id:<10} | {torch.log(torch.tensor(p_apple)).item():.4f}")

if __name__ == "__main__":
    main()


