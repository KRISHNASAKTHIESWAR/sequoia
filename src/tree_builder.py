# src/tree_builder.py

# #v1
# import torch

# class TreeBuilder:
#     def __init__(self, draft_model):
#         self.draft = draft_model

#     def build_simple_tree(self, input_ids):
#         """
#         Builds a Width=2, Depth=1 tree.
#         The Assistant gives its top 2 distinct guesses for the next token.
#         """
#         with torch.no_grad():
#             outputs = self.draft.model(input_ids)
#             # We only care about the predictions for the very last token
#             next_token_logits = outputs.logits[0, -1, :]
            
#             # Get the top 2 guesses. Shape becomes [1, 2]
#             top_2_guesses = torch.topk(next_token_logits, 2).indices.unsqueeze(0)
            
#         return top_2_guesses

# src/tree_builder.py
# import torch

# class TreeBuilder:
#     def __init__(self, draft_model):
#         self.draft = draft_model

#     def build_simple_tree(self, input_ids):
#         """
#         Builds a Width=2, Depth=1 tree.
#         The Assistant gives its top 2 distinct guesses for the next token.
#         """
#         with torch.no_grad():
#             outputs = self.draft.model(input_ids)
#             # We only care about the predictions for the very last token
#             next_token_logits = outputs.logits[0, -1, :]
            
#             # Get the top 2 guesses. Shape becomes [1, 2]
#             top_2_guesses = torch.topk(next_token_logits, 2).indices.unsqueeze(0)
            
#         return top_2_guesses

#v3 - KV cache

# src/tree_builder.py
# import torch

# class TreeBuilder:
#     def __init__(self, draft_model):
#         self.draft = draft_model

#     def build_simple_tree(self, last_token_id, past_key_values=None):
#         """
#         Builds a Width=2, Depth=1 tree using KV cache.
#         Processes ONLY the single last token to update its cache.
#         """
#         with torch.no_grad():
#             outputs = self.draft.model(
#                 input_ids=last_token_id, 
#                 past_key_values=past_key_values,
#                 use_cache=True
#             )
#             # Get logits for the single input token
#             next_token_logits = outputs.logits[0, -1, :]
            
#             # Sample without replacement: Top 2 distinct choices
#             top_2_guesses = torch.topk(next_token_logits, 2).indices.unsqueeze(0)
            
#         return top_2_guesses, outputs.past_key_values

#v4 - KV cache cloning - for depth=2 tree
# src/tree_builder.py
# src/tree_builder.py
# import torch
# import copy

# class TreeBuilder:
#     def __init__(self, draft_model):
#         self.draft = draft_model

#     def build_depth_2_tree(self, last_accepted_tokens, draft_past):
#         """
#         Builds a Depth=2 cascading tree using sequential batch-size-1 operations.
#         Catches up the assistant's cache with the newly accepted tokens before branching.
#         """
#         with torch.no_grad():
#             # 1. Catch up the Assistant's cache with the tokens verified in the last round
#             outputs = self.draft.model(input_ids=last_accepted_tokens, past_key_values=draft_past, use_cache=True)
#             draft_past_updated = outputs.past_key_values
            
#             # 2. Get predictions for the next token step
#             next_token_logits = outputs.logits[0, -1, :]
#             top_2_guesses = torch.topk(next_token_logits, 2).indices
            
#             guess_A = top_2_guesses[0].unsqueeze(0).unsqueeze(0) # Shape: [1, 1]
#             guess_B = top_2_guesses[1].unsqueeze(0).unsqueeze(0) # Shape: [1, 1]
            
#             # 3. Explore Branch A using a deep-cloned cache checkpoint
#             past_A = copy.deepcopy(draft_past_updated)
#             outputs_A = self.draft.model(input_ids=guess_A, past_key_values=past_A, use_cache=True)
#             guess_A_sub = torch.argmax(outputs_A.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
#             # 4. Explore Branch B using a deep-cloned cache checkpoint
#             past_B = copy.deepcopy(draft_past_updated)
#             outputs_B = self.draft.model(input_ids=guess_B, past_key_values=past_B, use_cache=True)
#             guess_B_sub = torch.argmax(outputs_B.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
#         return guess_A, guess_A_sub, guess_B, guess_B_sub, draft_past_updated
    
#v5 Tree sampling with temperature 
# src/tree_builder.py
# import torch
# import copy

# class TreeBuilder:
#     def __init__(self, draft_model):
#         self.draft = draft_model

#     def build_depth_2_tree(self, last_accepted_tokens, draft_past, temperature=0.7):
#         """
#         Builds a Depth=2 cascading tree using Stochastic Sampling Without Replacement.
#         """
#         with torch.no_grad():
#             outputs = self.draft.model(input_ids=last_accepted_tokens, past_key_values=draft_past, use_cache=True)
#             draft_past_updated = outputs.past_key_values
            
#             # --- STOCHASTIC SAMPLING: DEPTH 1 ---
#             next_token_logits = outputs.logits[0, -1, :]
            
#             # Apply Temperature to soften the probabilities (makes it creative)
#             scaled_logits = next_token_logits / temperature
#             probs = torch.softmax(scaled_logits, dim=-1)
            
#             # SEQUOIA UPGRADE: Sample 2 distinct tokens WITHOUT replacement
#             # This guarantees Branch A and Branch B are always exploring unique ideas
#             top_2_guesses = torch.multinomial(probs, num_samples=2, replacement=False)
            
#             guess_A = top_2_guesses[0].unsqueeze(0).unsqueeze(0) 
#             guess_B = top_2_guesses[1].unsqueeze(0).unsqueeze(0) 
            
#             # --- STOCHASTIC SAMPLING: DEPTH 2 (Branch A) ---
#             past_A = copy.deepcopy(draft_past_updated)
#             outputs_A = self.draft.model(input_ids=guess_A, past_key_values=past_A, use_cache=True)
#             probs_A = torch.softmax(outputs_A.logits[0, -1, :] / temperature, dim=-1)
#             guess_A_sub = torch.multinomial(probs_A, num_samples=1) # Roll for next word
            
#             # --- STOCHASTIC SAMPLING: DEPTH 2 (Branch B) ---
#             past_B = copy.deepcopy(draft_past_updated)
#             outputs_B = self.draft.model(input_ids=guess_B, past_key_values=past_B, use_cache=True)
#             probs_B = torch.softmax(outputs_B.logits[0, -1, :] / temperature, dim=-1)
#             guess_B_sub = torch.multinomial(probs_B, num_samples=1)
            
#         return guess_A, guess_A_sub, guess_B, guess_B_sub, draft_past_updated

#v6 dp
# src/tree_builder.py OPT
# import torch
# import copy

# class TreeBuilder:
#     def __init__(self, draft_model):
#         self.draft = draft_model

#     def build_dynamic_tree(self, mode, last_accepted_tokens, draft_past, temperature=0.7):
#         """The Traffic Cop: Routes to the correct tree shape math."""
#         if mode == "wide":
#             return self.build_wide_tree(last_accepted_tokens, draft_past, temperature)
#         elif mode == "deep":
#             return self.build_deep_tree(last_accepted_tokens, draft_past, temperature)
#         else:
#             return self.build_balanced_tree(last_accepted_tokens, draft_past, temperature)

#     def build_wide_tree(self, last_accepted_tokens, draft_past, temperature):
#         """WIDTH=4, DEPTH=1: Casts a wide net of 4 alternative next words."""
#         with torch.no_grad():
#             outputs = self.draft.model(input_ids=last_accepted_tokens, past_key_values=draft_past, use_cache=True)
#             draft_past_updated = outputs.past_key_values
            
#             probs = torch.softmax(outputs.logits[0, -1, :] / temperature, dim=-1)
#             # Sample 4 distinct guesses without replacement
#             top_4_guesses = torch.multinomial(probs, num_samples=4, replacement=False)
            
#             guesses = [top_4_guesses[i].unsqueeze(0).unsqueeze(0) for i in range(4)]
            
#         return guesses, draft_past_updated

#     def build_balanced_tree(self, last_accepted_tokens, draft_past, temperature):
#         """WIDTH=2, DEPTH=2: The balanced risk/reward tree."""
#         with torch.no_grad():
#             outputs = self.draft.model(input_ids=last_accepted_tokens, past_key_values=draft_past, use_cache=True)
#             draft_past_updated = outputs.past_key_values
            
#             probs = torch.softmax(outputs.logits[0, -1, :] / temperature, dim=-1)
#             top_2_guesses = torch.multinomial(probs, num_samples=2, replacement=False)
            
#             guess_A = top_2_guesses[0].unsqueeze(0).unsqueeze(0) 
#             guess_B = top_2_guesses[1].unsqueeze(0).unsqueeze(0) 
            
#             past_A = copy.deepcopy(draft_past_updated)
#             out_A = self.draft.model(input_ids=guess_A, past_key_values=past_A, use_cache=True)
#             guess_A_sub = torch.multinomial(torch.softmax(out_A.logits[0, -1, :] / temperature, dim=-1), 1)
            
#             past_B = copy.deepcopy(draft_past_updated)
#             out_B = self.draft.model(input_ids=guess_B, past_key_values=past_B, use_cache=True)
#             guess_B_sub = torch.multinomial(torch.softmax(out_B.logits[0, -1, :] / temperature, dim=-1), 1)
            
#         return (guess_A, guess_A_sub, guess_B, guess_B_sub), draft_past_updated

#     def build_deep_tree(self, last_accepted_tokens, draft_past, temperature):
#         """WIDTH=1, DEPTH=4: A straight line sprint into the future."""
#         with torch.no_grad():
#             outputs = self.draft.model(input_ids=last_accepted_tokens, past_key_values=draft_past, use_cache=True)
#             draft_past_updated = outputs.past_key_values
            
#             # Predict 4 tokens in a row
#             chain = []
#             curr_past = copy.deepcopy(draft_past_updated)
#             curr_token = torch.multinomial(torch.softmax(outputs.logits[0, -1, :]/temperature, dim=-1), 1)
#             chain.append(curr_token)
            
#             for _ in range(3):
#                 out = self.draft.model(input_ids=curr_token, past_key_values=curr_past, use_cache=True)
#                 curr_past = out.past_key_values
#                 curr_token = torch.multinomial(torch.softmax(out.logits[0, -1, :]/temperature, dim=-1), 1)
#                 chain.append(curr_token)
                
#         return chain, draft_past_updated

#v6 dp
# src/tree_builder.py Qwen
import torch
import copy

class TreeBuilder:
    def __init__(self, draft_model):
        self.draft = draft_model

    def _get_position_ids(self, past_key_values, input_ids):
        """Helper to calculate explicit position_ids to fix Qwen RoPE dimension crashes."""
        if past_key_values is None:
            seq_len = 0
        elif hasattr(past_key_values, "get_seq_length"):
            # Handles modern Hugging Face DynamicCache objects
            seq_len = past_key_values.get_seq_length()
        else:
            # Fallback for standard tuple cache format
            seq_len = past_key_values[0][0].shape[-2]
            
        # Safely determine token length (handling both 1D and 2D tensor shapes)
        token_len = input_ids.shape[1] if len(input_ids.shape) > 1 else input_ids.shape[0]
        
        # Create position IDs tracking from current sequence length forward
        pos_ids = torch.arange(seq_len, seq_len + token_len, dtype=torch.long, device=input_ids.device)
        return pos_ids.unsqueeze(0) # Shape: [1, token_len]

    def build_dynamic_tree(self, mode, last_accepted_tokens, draft_past, temperature=0.7):
        """The Traffic Cop: Routes to the correct tree shape math."""
        if mode == "wide":
            return self.build_wide_tree(last_accepted_tokens, draft_past, temperature)
        elif mode == "deep":
            return self.build_deep_tree(last_accepted_tokens, draft_past, temperature)
        else:
            return self.build_balanced_tree(last_accepted_tokens, draft_past, temperature)

    def build_wide_tree(self, last_accepted_tokens, draft_past, temperature):
        """WIDTH=4, DEPTH=1: Casts a wide net of 4 alternative next words."""
        with torch.no_grad():
            pos_ids = self._get_position_ids(draft_past, last_accepted_tokens)
            outputs = self.draft.model(input_ids=last_accepted_tokens, past_key_values=draft_past, position_ids=pos_ids, use_cache=True)
            draft_past_updated = outputs.past_key_values
            
            probs = torch.softmax(outputs.logits[0, -1, :] / temperature, dim=-1)
            top_4_guesses = torch.multinomial(probs, num_samples=4, replacement=False)
            
            guesses = [top_4_guesses[i].unsqueeze(0).unsqueeze(0) for i in range(4)]
            
        return guesses, draft_past_updated

    def build_balanced_tree(self, last_accepted_tokens, draft_past, temperature):
        """WIDTH=2, DEPTH=2: The balanced risk/reward tree."""
        with torch.no_grad():
            pos_ids = self._get_position_ids(draft_past, last_accepted_tokens)
            outputs = self.draft.model(input_ids=last_accepted_tokens, past_key_values=draft_past, position_ids=pos_ids, use_cache=True)
            draft_past_updated = outputs.past_key_values
            
            probs = torch.softmax(outputs.logits[0, -1, :] / temperature, dim=-1)
            top_2_guesses = torch.multinomial(probs, num_samples=2, replacement=False)
            
            guess_A = top_2_guesses[0].unsqueeze(0).unsqueeze(0) 
            guess_B = top_2_guesses[1].unsqueeze(0).unsqueeze(0) 
            
            # Sub-branch A
            past_A = copy.deepcopy(draft_past_updated)
            pos_ids_A = self._get_position_ids(past_A, guess_A)
            out_A = self.draft.model(input_ids=guess_A, past_key_values=past_A, position_ids=pos_ids_A, use_cache=True)
            guess_A_sub = torch.multinomial(torch.softmax(out_A.logits[0, -1, :] / temperature, dim=-1), 1).unsqueeze(0)
            
            # Sub-branch B
            past_B = copy.deepcopy(draft_past_updated)
            pos_ids_B = self._get_position_ids(past_B, guess_B)
            out_B = self.draft.model(input_ids=guess_B, past_key_values=past_B, position_ids=pos_ids_B, use_cache=True)
            guess_B_sub = torch.multinomial(torch.softmax(out_B.logits[0, -1, :] / temperature, dim=-1), 1).unsqueeze(0)
            
        return (guess_A, guess_A_sub, guess_B, guess_B_sub), draft_past_updated

    def build_deep_tree(self, last_accepted_tokens, draft_past, temperature):
        """WIDTH=1, DEPTH=4: A straight line sprint into the future."""
        with torch.no_grad():
            pos_ids = self._get_position_ids(draft_past, last_accepted_tokens)
            outputs = self.draft.model(input_ids=last_accepted_tokens, past_key_values=draft_past, position_ids=pos_ids, use_cache=True)
            draft_past_updated = outputs.past_key_values
            
            chain = []
            curr_past = copy.deepcopy(draft_past_updated)
            
            # Added .unsqueeze(0) to ensure curr_token is explicitly 2D [1, 1] instead of 1D [1]
            curr_token = torch.multinomial(torch.softmax(outputs.logits[0, -1, :]/temperature, dim=-1), 1).unsqueeze(0)
            chain.append(curr_token)
            
            for _ in range(3):
                pos_ids_curr = self._get_position_ids(curr_past, curr_token)
                out = self.draft.model(input_ids=curr_token, past_key_values=curr_past, position_ids=pos_ids_curr, use_cache=True)
                curr_past = out.past_key_values
                
                # Maintain 2D tensor structure dynamically
                curr_token = torch.multinomial(torch.softmax(out.logits[0, -1, :]/temperature, dim=-1), 1).unsqueeze(0)
                chain.append(curr_token)
                
        return chain, draft_past_updated