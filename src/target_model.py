from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TargetModel:
    def __init__(self, model_name="facebook/opt-125m"):
        print(f"Loading Target Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use CPU if GPU memory is too constrained, but try CUDA first
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set to evaluation mode
    
    def generate_baseline(self, prompt, max_new_tokens=50):
        """Generates text autoregressively (the slow way) to establish a baseline."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)