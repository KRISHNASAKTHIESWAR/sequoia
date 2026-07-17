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
# import time
# import torch
# from src.target_model import TargetModel
# from src.draft_model import DraftModel
# from src.tree_builder import TreeBuilder
# from src.verifier import TreeVerifier

# def run_sequoia_kv_cache():
#     print("==================================================")
#     print("STEP 1: LOADING MODELS INTO RAM")
#     print("==================================================")
#     target = TargetModel("facebook/opt-350m")  
#     draft = DraftModel("facebook/opt-125m")    
    
#     tree_builder = TreeBuilder(draft)
#     verifier = TreeVerifier(target)
    
#     prompt = "The future of artificial intelligence is very"
#     print(f"\nPrompt: {prompt}")
    
#     input_ids = target.tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)
    
#     print("\n==================================================")
#     print("STEP 2: PRE-POPULATION PHASE (INITIALIZING KV CACHE)")
#     print("==================================================")
#     # Process the whole prompt once to generate initial cache folders
#     with torch.no_grad():
#         target_outputs = target.model(input_ids, use_cache=True)
#         target_past = target_outputs.past_key_values
        
#         draft_outputs = draft.model(input_ids, use_cache=True)
#         draft_past = draft_outputs.past_key_values
    
#     # Get the first token prediction coming right out of the prompt
#     next_token_id = torch.argmax(target_outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
#     input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    
#     max_new_tokens = 50
#     tokens_generated = 1  # We already got the first token from pre-pop!
    
#     print("\n==================================================")
#     print("STEP 3: HYPERSPEED SEQUOIA LOOP")
#     print("==================================================")
#     start_time = time.time()
    
#     # Track the single latest token to feed into the models
#     current_token_to_process = next_token_id
    
#     while tokens_generated < max_new_tokens:
#         # A. Assistant predicts guesses for the next step & updates its cache
#         tree_tokens, draft_past = tree_builder.build_simple_tree(current_token_to_process, past_key_values=draft_past)
        
#         # B. Architect verifies the guesses & updates its cache
#         accepted_token, target_past = verifier.verify_tree(current_token_to_process, tree_tokens, past_key_values=target_past)
        
#         # C. Stitch the accepted token into our full final text tracker
#         input_ids = torch.cat([input_ids, accepted_token], dim=-1)
        
#         # Advance loop state
#         current_token_to_process = accepted_token
#         tokens_generated += 1
        
#         if tokens_generated % 10 == 0:
#             print(f"-> Progress: Generated {tokens_generated}/{max_new_tokens} tokens...")

#     end_time = time.time()
    
#     final_output = target.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
#     print("\n==================================================")
#     print("STEP 4: FINAL RESULTS WITH KV CACHE")
#     print("==================================================")
#     print(f"--- Final Sequoia Output ---\n{final_output}")
#     print(f"\nTime taken for {max_new_tokens} tokens: {end_time - start_time:.2f} seconds")

# if __name__ == "__main__":
#     run_sequoia_kv_cache()

#v4 - KV cache cloning for depth=2 tree
# main.py
# main.py (Final Polish with Analytics)
# import time
# import torch
# from src.target_model import TargetModel
# from src.draft_model import DraftModel
# from src.tree_builder import TreeBuilder
# from src.verifier import TreeVerifier

# def run_sequoia_depth_2():
#     print("==================================================")
#     print("STEP 1: LOADING MODELS INTO RAM")
#     print("==================================================")
#     target = TargetModel("facebook/opt-350m")  
#     draft = DraftModel("facebook/opt-125m")    
    
#     tree_builder = TreeBuilder(draft)
#     verifier = TreeVerifier(target)
    
#     prompt = "The future of artificial intelligence is very"
#     print(f"\nPrompt: {prompt}")
    
#     input_ids = target.tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)
    
#     print("\n==================================================")
#     print("STEP 2: PRE-POPULATION PHASE")
#     print("==================================================")
#     with torch.no_grad():
#         target_outputs = target.model(input_ids, use_cache=True)
#         target_past = target_outputs.past_key_values
        
#         draft_outputs = draft.model(input_ids, use_cache=True)
#         draft_past = draft_outputs.past_key_values
    
#     first_token = torch.argmax(target_outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
#     input_ids = torch.cat([input_ids, first_token], dim=-1)
    
#     max_new_tokens = 50
#     tokens_generated = 1
    
#     # --- METRIC TRACKERS ---
#     double_wins = 0
#     single_wins = 0
#     misses = 0
#     total_steps = 0
    
#     print("\n==================================================")
#     print("STEP 3: RUNNING CASCADING DEPTH=2 TREE LOOP")
#     print("==================================================")
#     start_time = time.time()
    
#     last_accepted_tokens = first_token
    
#     while tokens_generated < max_new_tokens:
#         total_steps += 1
        
#         # A. Assistant builds guesses
#         guess_A, guess_A_sub, guess_B, guess_B_sub, draft_past = tree_builder.build_depth_2_tree(
#             last_accepted_tokens, draft_past
#         )
        
#         # B. Architect validates guesses
#         accepted_tokens, target_past = verifier.verify_depth_2_tree(
#             last_accepted_tokens, guess_A, guess_A_sub, guess_B, guess_B_sub, target_past
#         )
        
#         # Track statistics based on token length returned
#         num_accepted = accepted_tokens.shape[1]
#         if num_accepted == 2:
#             double_wins += 1
#         elif num_accepted == 1 and "Correction" in verifier.__class__.__name__ or num_accepted == 1:
#             # Simple fallback counter tracking
#             if "failed completely" in open('main.py').read(): # Placeholder logic check
#                 pass
        
#         # C. Stitch tokens
#         input_ids = torch.cat([input_ids, accepted_tokens], dim=-1)
        
#         tokens_generated += num_accepted
#         last_accepted_tokens = accepted_tokens
        
#         print(f"     Progress: {tokens_generated}/{max_new_tokens} tokens tracked.")

#     end_time = time.time()
#     final_output = target.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
#     # --- CALCULATION OF METRICS ---
#     avg_tokens_per_step = tokens_generated / total_steps
    
#     print("\n==================================================")
#     print("STEP 4: FINAL RESEARCH METRICS REPORT")
#     print("==================================================")
#     print(f"--- Final Sequoia Output ---\n{final_output}\n")
#     print(f"Total Alpha-Steps Taken: {total_steps}")
#     print(f"Average Tokens Generated Per Step: {avg_tokens_per_step:.2f}x speedup capability")
#     print(f"Wall-clock Time: {end_time - start_time:.2f} seconds")
#     print("==================================================")

# if __name__ == "__main__":
#     run_sequoia_depth_2()

#v6 dp with zipfian vocabulary pruning

# main.py
import time
import torch
from collections import deque
from src.target_model import TargetModel
from src.draft_model import DraftModel
from src.tree_builder import TreeBuilder
from src.verifier import TreeVerifier
from src.dp_optimizer import SequoiaDPOptimizer

def run_sequoia_interactive():
    print("==================================================")
    print("STEP 1: LOADING MODELS & OPTIMIZER")
    print("==================================================")
    # target = TargetModel("facebook/opt-350m")  
    # draft = DraftModel("facebook/opt-125m", vocab_size=10000)   

    # Target (Architect): 1.5 Billion parameters (Smart, Instruct-Tuned)
    target = TargetModel("Qwen/Qwen2.5-1.5B-Instruct")  
    
    # Draft (Assistant): 0.5 Billion parameters (Fast, Instruct-Tuned)
    # We keep vocab_size=10000 for your Zipfian Pruning!
    draft = DraftModel("Qwen/Qwen2.5-0.5B-Instruct", vocab_size=10000) 
    
    tree_builder = TreeBuilder(draft)
    verifier = TreeVerifier(target)
    optimizer = SequoiaDPOptimizer(compute_budget=4)
    
    print("\n==================================================")
    print("SYSTEM READY. TYPE 'exit' TO QUIT.")
    print("==================================================")
    
    # --- INTERACTIVE PROMPT LOOP ---
    while True:
        try:
            prompt = input("\n[USER]: ")
            if prompt.lower() in ["exit", "quit", "stop"]:
                print("Exiting Sequoia Engine. Goodbye!")
                break
            if not prompt.strip():
                continue
                
        except KeyboardInterrupt:
            # Safely catch Ctrl+C
            print("\nExiting Sequoia Engine. Goodbye!")
            break
            
        print("[SEQUOIA]: Generating...")
        
    
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        
        # Tokenize the wrapped prompt instead of the raw user text
        input_ids = target.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(target.device)
        
        # Pre-population Phase
        with torch.no_grad():
            target_outputs = target.model(input_ids, use_cache=True)
            target_past = target_outputs.past_key_values
            draft_outputs = draft.model(input_ids, use_cache=True)
            draft_past = draft_outputs.past_key_values
        
        first_token = torch.argmax(target_outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, first_token], dim=-1)
        
        # Reset generation trackers for the new prompt
        max_new_tokens = 50
        tokens_generated = 1
        total_steps = 0
        accuracy_queue = deque([0.6] * 10, maxlen=10)
        
        start_time = time.time()
        last_accepted_tokens = first_token
        
        # The Dynamic Tree Loop
        while tokens_generated < max_new_tokens:
            total_steps += 1
            
            current_acc = sum(accuracy_queue) / len(accuracy_queue)
            mode = optimizer.find_optimal_shape(current_acc)
            
            tree_data, draft_past = tree_builder.build_dynamic_tree(mode, last_accepted_tokens, draft_past)
            accepted_tokens, target_past = verifier.verify_dynamic_tree(mode, last_accepted_tokens, tree_data, target_past)
            
            num_accepted = accepted_tokens.shape[1]
            step_accuracy = 1.0 if num_accepted > 1 else 0.0
            accuracy_queue.append(step_accuracy)
            
            input_ids = torch.cat([input_ids, accepted_tokens], dim=-1)
            tokens_generated += num_accepted
            last_accepted_tokens = accepted_tokens
            
            # Print intermediate progress (Optional: You can comment this out for a cleaner UI)
            # print(f"[{mode.upper()}] Step {total_steps} | Accuracy: {current_acc*100:.0f}% | Tokens: {tokens_generated}/{max_new_tokens}")

        end_time = time.time()
        final_output = target.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Print the final generated block with its metrics
        print(f"\n{final_output}")
        print("-" * 50)
        print(f"Metrics: {total_steps} steps | Speedup: {tokens_generated / total_steps:.2f}x | Time: {end_time - start_time:.2f}s")
        print("-" * 50)

if __name__ == "__main__":
    run_sequoia_interactive()