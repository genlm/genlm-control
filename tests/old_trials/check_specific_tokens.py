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
    
    # banana tokens
    # banana | [3820, 2271]
    banana_first_token = 3820
    
    # apple token
    apple_token = 18040
    
    # chars
    a_token = 64
    b_token = 65

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

        p_ban = probs[banana_first_token].item()
        p_apple = probs[apple_token].item()
        p_a = probs[a_token].item()
        p_b = probs[b_token].item()
        
        log_ban = torch.log(torch.tensor(p_ban)).item()
        log_apple = torch.log(torch.tensor(p_apple)).item()
        log_a = torch.log(torch.tensor(p_a)).item()
        log_b = torch.log(torch.tensor(p_b)).item()
        
        print(f"{'Token':<10} | {'ID':<10} | {'LogProb':<10}")
        print("-" * 40)
        print(f"{'ban':<10} | {banana_first_token:<10} | {log_ban:.4f}")
        print(f"{'apple':<10} | {apple_token:<10} | {log_apple:.4f}")
        print(f"{'a':<10} | {a_token:<10} | {log_a:.4f}")
        print(f"{'b':<10} | {b_token:<10} | {log_b:.4f}")

if __name__ == "__main__":
    main()


