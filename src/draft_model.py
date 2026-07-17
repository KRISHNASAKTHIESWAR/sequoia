# src/draft_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class DraftModel:
    def __init__(self, model_name="facebook/opt-125m", vocab_size=10000):
        print(f"Loading Draft Model: {model_name}...")
        self.device = "cpu" # or cuda if available
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # ==========================================================
        # VOCABULARY PRUNING (SPEC-VOCAB)
        # ==========================================================
        # 1. Grab the original massive output layer
        old_head = self.model.lm_head
        
        # 2. Create a smaller, faster output layer
        print(f"-> Pruning Draft Vocabulary from {old_head.out_features} down to {vocab_size} tokens...")
        new_head = torch.nn.Linear(old_head.in_features, vocab_size, bias=False).to(self.device)
        
        # 3. Copy only the weights for the top 'vocab_size' most common tokens
        with torch.no_grad():
            new_head.weight.data = old_head.weight.data[:vocab_size, :].clone()
            
        # 4. Surgically replace the Assistant's brain with the fast head
        self.model.lm_head = new_head