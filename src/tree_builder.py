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
import torch

class TreeBuilder:
    def __init__(self, draft_model):
        self.draft = draft_model

    def build_simple_tree(self, last_token_id, past_key_values=None):
        """
        Builds a Width=2, Depth=1 tree using KV cache.
        Processes ONLY the single last token to update its cache.
        """
        with torch.no_grad():
            outputs = self.draft.model(
                input_ids=last_token_id, 
                past_key_values=past_key_values,
                use_cache=True
            )
            # Get logits for the single input token
            next_token_logits = outputs.logits[0, -1, :]
            
            # Sample without replacement: Top 2 distinct choices
            top_2_guesses = torch.topk(next_token_logits, 2).indices.unsqueeze(0)
            
        return top_2_guesses, outputs.past_key_values