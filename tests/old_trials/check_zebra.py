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

    candidates = ["zebra", "zucchini", "apple", "mandarin", "mango", "melon"]
    
    print(f"{'Word':<15} | {'Tokens':<20} | {'LogProb (Sum)':<15} | {'First Token LogProb':<20}")
    print("-" * 80)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

        for word in candidates:
            # Encode `prompt + word` to get correct tokens
            full_ids = tokenizer.encode(prompt + word)
            word_ids = full_ids[len(input_ids[0]):]
            
            if not word_ids: continue
            
            # First token prob
            p1 = probs[word_ids[0]].item()
            log_p1 = torch.log(torch.tensor(p1)).item()
            
            # Total prob
            temp_ids = input_ids
            total_log_prob = 0.0
            for i, tid in enumerate(word_ids):
                out = model(temp_ids)
                l = out.logits[0, -1, :]
                p = F.softmax(l, dim=-1)[tid].item()
                total_log_prob += torch.log(torch.tensor(p)).item()
                temp_ids = torch.cat([temp_ids, torch.tensor([[tid]])], dim=1)

            print(f"{word:<15} | {str(word_ids):<20} | {total_log_prob:.4f}          | {log_p1:.4f}")
            
        # Check char 'z'
        z_id = tokenizer.encode("z")[0] # 89
        p_z = probs[z_id].item()
        print(f"\nChar 'z' (id {z_id}) LogProb: {torch.log(torch.tensor(p_z)).item():.4f}")
        
        # Check char 'm'
        m_id = tokenizer.encode("m")[0]
        p_m = probs[m_id].item()
        print(f"Char 'm' (id {m_id}) LogProb: {torch.log(torch.tensor(p_m)).item():.4f}")

if __name__ == "__main__":
    main()


