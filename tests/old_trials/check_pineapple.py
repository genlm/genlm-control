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

    candidates = ["pineapple", "apple"]
    
    print(f"{'Word':<15} | {'Tokens':<20} | {'LogProb (Sum)':<15} | {'First Token LogProb':<20}")
    print("-" * 80)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

        for word in candidates:
            full_ids = tokenizer.encode(prompt + word)
            word_ids = full_ids[len(input_ids[0]):]
            
            p1 = probs[word_ids[0]].item()
            log_p1 = torch.log(torch.tensor(p1)).item()
            
            print(f"{word:<15} | {str(word_ids):<20} | N/A               | {log_p1:.4f}")

if __name__ == "__main__":
    main()


