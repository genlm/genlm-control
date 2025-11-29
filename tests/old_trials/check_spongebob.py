import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    prompt = "SpongeBob SquarePants lives in a "
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    candidates = ["pineapple", "house"]
    
    print(f"{'Word':<15} | {'Tokens':<20} | {'First Token LogProb':<20}")
    print("-" * 60)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

        for word in candidates:
            # Tokenizer encode adds space if not present? No.
            # But " lives in a " ends with space.
            # 'pineapple' might be [23908, 18040] (pin, apple) OR [30998] (pineapple) if it exists?
            # Let's encode just the word to see.
            # And encode prompt + word to be sure.
            
            full_ids = tokenizer.encode(prompt + word)
            word_ids = full_ids[len(input_ids[0]):]
            
            # If word_ids is empty, something is wrong with logic
            if not word_ids:
                print(f"{word:<15} | EMPTY IDS")
                continue
            
            p1 = probs[word_ids[0]].item()
            log_p1 = torch.log(torch.tensor(p1)).item()
            
            print(f"{word:<15} | {str(word_ids):<20} | {log_p1:.4f}")

if __name__ == "__main__":
    main()
