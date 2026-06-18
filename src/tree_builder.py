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
import torch
import copy

class TreeBuilder:
    def __init__(self, draft_model):
        self.draft = draft_model

    def build_depth_2_tree(self, last_accepted_tokens, draft_past):
        """
        Builds a Depth=2 cascading tree using sequential batch-size-1 operations.
        Catches up the assistant's cache with the newly accepted tokens before branching.
        """
        with torch.no_grad():
            # 1. Catch up the Assistant's cache with the tokens verified in the last round
            outputs = self.draft.model(input_ids=last_accepted_tokens, past_key_values=draft_past, use_cache=True)
            draft_past_updated = outputs.past_key_values
            
            # 2. Get predictions for the next token step
            next_token_logits = outputs.logits[0, -1, :]
            top_2_guesses = torch.topk(next_token_logits, 2).indices
            
            guess_A = top_2_guesses[0].unsqueeze(0).unsqueeze(0) # Shape: [1, 1]
            guess_B = top_2_guesses[1].unsqueeze(0).unsqueeze(0) # Shape: [1, 1]
            
            # 3. Explore Branch A using a deep-cloned cache checkpoint
            past_A = copy.deepcopy(draft_past_updated)
            outputs_A = self.draft.model(input_ids=guess_A, past_key_values=past_A, use_cache=True)
            guess_A_sub = torch.argmax(outputs_A.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
            # 4. Explore Branch B using a deep-cloned cache checkpoint
            past_B = copy.deepcopy(draft_past_updated)
            outputs_B = self.draft.model(input_ids=guess_B, past_key_values=past_B, use_cache=True)
            guess_B_sub = torch.argmax(outputs_B.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
        return guess_A, guess_A_sub, guess_B, guess_B_sub, draft_past_updated