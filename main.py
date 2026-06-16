# v2
# import time
# import torch
# from src.target_model import TargetModel
# from src.draft_model import DraftModel
# from src.tree_builder import TreeBuilder
# from src.verifier import TreeVerifier

# def run_sequoia_prototype():
#     print("==================================================")
#     print("STEP 1: LOADING MODELS INTO RAM (ONLY HAPPENS ONCE)")
#     print("==================================================")
    
#     # Load the models ONCE at the start of the script
#     target = TargetModel("facebook/opt-350m")  
#     draft = DraftModel("facebook/opt-125m")    
    
#     # Initialize the Sequoia tree builder and verifier ONCE
#     tree_builder = TreeBuilder(draft)
#     verifier = TreeVerifier(target)
    
#     prompt = "The future of artificial intelligence is very"
#     print(f"\nPrompt: {prompt}")
    
#     # Convert text prompt to token IDs
#     input_ids = target.tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)
    
#     max_new_tokens = 50
#     tokens_generated = 0
    
#     print("\n==================================================")
#     print("STEP 2: STARTING THE SEQUOIA GENERATION LOOP")
#     print("==================================================")
#     start_time = time.time()
    
#     # The loop ONLY does the token processing math. 
#     # NO models are re-loaded or re-downloaded inside this block.
#     while tokens_generated < max_new_tokens:
#         # A. Assistant guesses tokens
#         tree_tokens = tree_builder.build_simple_tree(input_ids)
        
#         # B. Architect verifies the guesses
#         accepted_token = verifier.verify_tree(input_ids, tree_tokens)
        
#         # C. Append the accepted token to our sequence
#         input_ids = torch.cat([input_ids, accepted_token], dim=-1)
        
#         tokens_generated += 1
        
#         if tokens_generated % 10 == 0:
#             print(f"-> Progress: Generated {tokens_generated}/{max_new_tokens} tokens...")

#     end_time = time.time()
    
#     # Decode and print the final sequence
#     final_output = target.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
#     print("\n==================================================")
#     print("STEP 3: RESULTS")
#     print("==================================================")
#     print(f"--- Final Sequoia Output ---\n{final_output}")
#     print(f"\nTime taken for {max_new_tokens} tokens: {end_time - start_time:.2f} seconds")

# if __name__ == "__main__":
#     run_sequoia_prototype()

#v3 - KV cache
# main.py
import time
import torch
from src.target_model import TargetModel
from src.draft_model import DraftModel
from src.tree_builder import TreeBuilder
from src.verifier import TreeVerifier

def run_sequoia_kv_cache():
    print("==================================================")
    print("STEP 1: LOADING MODELS INTO RAM")
    print("==================================================")
    target = TargetModel("facebook/opt-350m")  
    draft = DraftModel("facebook/opt-125m")    
    
    tree_builder = TreeBuilder(draft)
    verifier = TreeVerifier(target)
    
    prompt = "The future of artificial intelligence is very"
    print(f"\nPrompt: {prompt}")
    
    input_ids = target.tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)
    
    print("\n==================================================")
    print("STEP 2: PRE-POPULATION PHASE (INITIALIZING KV CACHE)")
    print("==================================================")
    # Process the whole prompt once to generate initial cache folders
    with torch.no_grad():
        target_outputs = target.model(input_ids, use_cache=True)
        target_past = target_outputs.past_key_values
        
        draft_outputs = draft.model(input_ids, use_cache=True)
        draft_past = draft_outputs.past_key_values
    
    # Get the first token prediction coming right out of the prompt
    next_token_id = torch.argmax(target_outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    
    max_new_tokens = 50
    tokens_generated = 1  # We already got the first token from pre-pop!
    
    print("\n==================================================")
    print("STEP 3: HYPERSPEED SEQUOIA LOOP")
    print("==================================================")
    start_time = time.time()
    
    # Track the single latest token to feed into the models
    current_token_to_process = next_token_id
    
    while tokens_generated < max_new_tokens:
        # A. Assistant predicts guesses for the next step & updates its cache
        tree_tokens, draft_past = tree_builder.build_simple_tree(current_token_to_process, past_key_values=draft_past)
        
        # B. Architect verifies the guesses & updates its cache
        accepted_token, target_past = verifier.verify_tree(current_token_to_process, tree_tokens, past_key_values=target_past)
        
        # C. Stitch the accepted token into our full final text tracker
        input_ids = torch.cat([input_ids, accepted_token], dim=-1)
        
        # Advance loop state
        current_token_to_process = accepted_token
        tokens_generated += 1
        
        if tokens_generated % 10 == 0:
            print(f"-> Progress: Generated {tokens_generated}/{max_new_tokens} tokens...")

    end_time = time.time()
    
    final_output = target.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    print("\n==================================================")
    print("STEP 4: FINAL RESULTS WITH KV CACHE")
    print("==================================================")
    print(f"--- Final Sequoia Output ---\n{final_output}")
    print(f"\nTime taken for {max_new_tokens} tokens: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_sequoia_kv_cache()