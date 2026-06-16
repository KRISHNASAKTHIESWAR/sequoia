from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DraftModel:
    def __init__(self, model_name="JackFram/llama-68m"):
        print(f"Loading Draft Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()