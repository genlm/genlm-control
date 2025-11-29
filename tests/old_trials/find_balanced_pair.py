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

    candidates = [
        "apple", "apricot", "avocado", 
        "banana", "blackberry", "blueberry",
        "cherry", "coconut", "cranberry",
        "date", "dragonfruit",
        "elderberry",
        "fig",
        "grape", "grapefruit", "guava",
        "kiwi",
        "lemon", "lime", "lychee",
        "mango", "melon", "mandarin",
        "nectarine",
        "orange",
        "papaya", "peach", "pear", "plum", "pineapple", "pomegranate",
        "raspberry",
        "strawberry",
        "tangerine",
        "watermelon"
    ]
    
    # Also consider non-fruits if needed to balance
    candidates += ["pepper", "potato", "tomato", "carrot"]

    print(f"{'Word':<15} | {'Tokens':<20} | {'Log Prob (Sum)':<15} | {'First Token LogProb':<20}")
    print("-" * 80)

    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)

        for word in candidates:
            # Force no space
            # Note: tokenizer.encode(word) usually adds space if not specified, 
            # but GPT2 tokenizer is tricky. 
            # We want the tokens that would be generated after "["
            # So we encode `prompt + word` and subtract `prompt` ids
            full_ids = tokenizer.encode(prompt + word)
            word_ids = full_ids[len(input_ids[0]):]
            
            if not word_ids: continue

            # Calculate probability of this sequence
            # 1. First token prob
            p1 = probs[word_ids[0]].item()
            log_p1 = torch.log(torch.tensor(p1)).item()
            
            # 2. Total prob (approximate, by following greedy path? No, force path)
            curr_ids = input_ids
            total_log_prob = 0.0
            
            temp_ids = input_ids
            
            valid = True
            for i, tid in enumerate(word_ids):
                out = model(temp_ids)
                logits = out.logits[0, -1, :]
                p = F.softmax(logits, dim=-1)[tid].item()
                total_log_prob += torch.log(torch.tensor(p)).item()
                temp_ids = torch.cat([temp_ids, torch.tensor([[tid]])], dim=1)
                
                if i == 0:
                    first_token_log_prob = torch.log(torch.tensor(p)).item()

            print(f"{word:<15} | {str(word_ids):<20} | {total_log_prob:.4f}          | {first_token_log_prob:.4f}")

if __name__ == "__main__":
    main()


