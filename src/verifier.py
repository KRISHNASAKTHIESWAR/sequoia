#v1
# import torch

# class TreeVerifier:
#     def __init__(self, target_model):
#         self.target = target_model

#     def verify_tree(self, input_ids, tree_tokens):
#         """
#         Takes the tree and verifies it in ONE forward pass.
#         """
#         # THE MAGIC TRICK: Topology-Aware Causal Mask
#         # You must create a custom 2D attention mask matrix. 
#         # If Token B is a branch off Token A, the mask must allow B to look at A, 
#         # but prevent B from looking at Token C (which is on a parallel branch).
        
#         # Flatten the tree into a single sequence
#         flattened_tree_ids = self._flatten_tree(tree_tokens)
#         combined_inputs = torch.cat([input_ids, flattened_tree_ids], dim=-1)
        
#         # TODO: Create custom attention mask here
#         # custom_mask = ... 

#         with torch.no_grad():
#             # Pass everything through the Architect at once
#             outputs = self.target.model(combined_inputs) # Pass mask here
            
#         # Acceptance Logic (Greedy)
#         # Check the Target model's logits against the flattened_tree_ids.
#         # Find the longest continuous branch where the Target's top choice 
#         # matches the Draft's guess.
        
#         accepted_tokens = [] # List of tokens the architect agreed with
#         return accepted_tokens
        
#     def _flatten_tree(self, tree):
#         # Helper function to turn the tree structure into a 1D tensor
#         pass

# src/verifier.py
# import torch

# class TreeVerifier:
#     def __init__(self, target_model):
#         self.target = target_model

#     def verify_tree(self, input_ids, tree_tokens):
#         """
#         Verifies both draft guesses simultaneously using a single batch pass.
#         This completely bypasses Hugging Face's strict internal mask constraints.
#         """
#         seq_len = input_ids.shape[1]
        
#         # 1. Split the top 2 guesses out from the tree_tokens tensor [1, 2]
#         guess_1 = tree_tokens[:, 0:1]  # Shape: [1, 1]
#         guess_2 = tree_tokens[:, 1:2]  # Shape: [1, 1]
        
#         # 2. Create two parallel sequences
#         seq_1 = torch.cat([input_ids, guess_1], dim=-1)
#         seq_2 = torch.cat([input_ids, guess_2], dim=-1)
        
#         # 3. Combine them into a single batch of size 2. Shape: [2, seq_len + 1]
#         batched_inputs = torch.cat([seq_1, seq_2], dim=0).to(self.target.device)
        
#         with torch.no_grad():
#             # Pass the batch through the Architect in ONE single forward pass
#             outputs = self.target.model(batched_inputs)
            
#         # 4. Get the Architect's prediction for the token right after the prompt.
#         # The prompt ends at index `seq_len - 1`. The prediction for what comes next
#         # is located at index `seq_len - 1` in the logits matrix.
#         architect_logits = outputs.logits[0, seq_len - 1, :]
#         architect_top_choice = torch.argmax(architect_logits)
        
#         # 5. Acceptance Logic (.item() avoids cross-device tensor issues)
#         g1_token = tree_tokens[0, 0].item()
#         g2_token = tree_tokens[0, 1].item()
#         actual_token = architect_top_choice.item()
        
#         if actual_token == g1_token:
#             word = self.target.tokenizer.decode(g1_token)
#             print(f"  -> Architect accepted Guess 1: '{word}'")
#             return tree_tokens[0, 0].unsqueeze(0).unsqueeze(0)
            
#         elif actual_token == g2_token:
#             word = self.target.tokenizer.decode(g2_token)
#             print(f"  -> Architect accepted Guess 2: '{word}'")
#             return tree_tokens[0, 1].unsqueeze(0).unsqueeze(0)
            
#         else:
#             word = self.target.tokenizer.decode(actual_token)
#             print(f"  -> Architect rejected both. Correcting to: '{word}'")
#             return architect_top_choice.unsqueeze(0).unsqueeze(0)

#v3 - KV cache
# src/verifier.py
import torch

class TreeVerifier:
    def __init__(self, target_model):
        self.target = target_model

    def verify_tree(self, last_token_id, tree_tokens, past_key_values=None):
        """
        Verifies the guesses using the Target Model's KV cache.
        Processes ONLY the single last token to update the cache and get the next choice.
        """
        with torch.no_grad():
            outputs = self.target.model(
                input_ids=last_token_id,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        # Get the Architect's true prediction for the next token
        architect_logits = outputs.logits[0, -1, :]
        architect_top_choice = torch.argmax(architect_logits)
        
        # Pull out token IDs for verification logs
        g1_token = tree_tokens[0, 0].item()
        g2_token = tree_tokens[0, 1].item()
        actual_token = architect_top_choice.item()
        
        if actual_token == g1_token:
            word = self.target.tokenizer.decode(g1_token)
            print(f"  -> Architect accepted Guess 1: '{word}'")
        elif actual_token == g2_token:
            word = self.target.tokenizer.decode(g2_token)
            print(f"  -> Architect accepted Guess 2: '{word}'")
        else:
            word = self.target.tokenizer.decode(actual_token)
            print(f"  -> Architect rejected both. Correcting to: '{word}'")
            
        # Return the target's true token and its updated cache folder
        return architect_top_choice.unsqueeze(0).unsqueeze(0), outputs.past_key_values