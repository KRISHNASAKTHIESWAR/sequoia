# src/dp_optimizer.py

#test_optimizer.py
# class SequoiaDPOptimizer:
#     def __init__(self, compute_budget=4):
#         """
#         compute_budget: The maximum number of extra tokens your GPU/CPU can 
#         evaluate in a single forward pass without slowing down.
#         """
#         self.budget = compute_budget

#     def expected_tokens(self, width, depth, p):
#         """
#         Calculates the expected number of accepted tokens for a specific tree shape.
#         p = The probability that the Assistant's guess is accepted by the Architect.
#         """
#         # Math: The probability that AT LEAST ONE branch at depth 1 is correct
#         prob_level_1 = 1 - (1 - p)**width
        
#         # If depth is 1, the maximum expected tokens is just the probability of level 1
#         if depth == 1:
#             return prob_level_1
            
#         # If depth is 2, it is the expected tokens of level 1, PLUS the expected 
#         # tokens of level 2 (which relies on level 1 being correct)
#         expected = prob_level_1 * (1 + p**(depth - 1))
#         return expected

#     def find_optimal_shape(self, current_accuracy_rate):
#         """
#         The Dynamic Programming Simulator: Tests all possible tree shapes 
#         within the compute budget to find the absolute maximum expected speedup.
#         """
#         p = current_accuracy_rate
#         best_shape = None
#         max_expected_tokens = 0.0

#         print(f"\n--- Running DP Optimizer (Assistant Accuracy: {p*100:.1f}%) ---")
        
#         # Test Shape 1: Wide & Shallow (Width=4, Depth=1)
#         # Good for low accuracy
#         exp_wide = self.expected_tokens(width=self.budget, depth=1, p=p)
#         print(f"Shape [Width=4, Depth=1] -> Expected Tokens: {exp_wide:.3f}")
#         if exp_wide > max_expected_tokens:
#             max_expected_tokens = exp_wide
#             best_shape = "Wide Bush (W=4, D=1)"

#         # Test Shape 2: The Balanced Tree (Width=2, Depth=2)
#         # Good for medium accuracy (This is your current codebase!)
#         exp_balanced = self.expected_tokens(width=2, depth=2, p=p)
#         print(f"Shape [Width=2, Depth=2] -> Expected Tokens: {exp_balanced:.3f}")
#         if exp_balanced > max_expected_tokens:
#             max_expected_tokens = exp_balanced
#             best_shape = "Balanced Tree (W=2, D=2)"

#         # Test Shape 3: Deep & Narrow (Width=1, Depth=4)
#         # Good for high accuracy
#         # Expected tokens for a straight line is the sum of probabilities: p + p^2 + p^3 + p^4
#         exp_deep = p + p**2 + p**3 + p**4
#         print(f"Shape [Width=1, Depth=4] -> Expected Tokens: {exp_deep:.3f}")
#         if exp_deep > max_expected_tokens:
#             max_expected_tokens = exp_deep
#             best_shape = "Straight Line (W=1, D=4)"

#         print(f">> OPTIMAL SELECTION: {best_shape}")
#         return best_shape

# src/dp_optimizer.py

class SequoiaDPOptimizer:
    def __init__(self, compute_budget=4):
        self.budget = compute_budget

    def expected_tokens(self, width, depth, p):
        # Probability that at least one branch is correct
        prob_level_1 = 1 - (1 - p)**width
        if depth == 1:
            return prob_level_1
        # Expected tokens for depth=2
        expected = prob_level_1 * (1 + p**(depth - 1))
        return expected

    def find_optimal_shape(self, current_accuracy_rate):
        p = current_accuracy_rate
        max_expected_tokens = 0.0
        best_mode = "balanced" # Default fallback

        # 1. Evaluate Wide Bush (Good for low accuracy)
        exp_wide = self.expected_tokens(width=4, depth=1, p=p)
        if exp_wide > max_expected_tokens:
            max_expected_tokens = exp_wide
            best_mode = "wide"

        # 2. Evaluate Balanced Tree (Good for medium accuracy)
        exp_balanced = self.expected_tokens(width=2, depth=2, p=p)
        if exp_balanced > max_expected_tokens:
            max_expected_tokens = exp_balanced
            best_mode = "balanced"

        # 3. Evaluate Deep Line (Good for high accuracy)
        exp_deep = p + p**2 + p**3 + p**4
        if exp_deep > max_expected_tokens:
            max_expected_tokens = exp_deep
            best_mode = "deep"

        return best_mode