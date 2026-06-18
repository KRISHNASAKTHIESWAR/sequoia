# test_optimizer.py
from src.dp_optimizer import SequoiaDPOptimizer

optimizer = SequoiaDPOptimizer(compute_budget=4)

# Scenario A: The AI is struggling with a complex math problem (Accuracy = 30%)
optimizer.find_optimal_shape(current_accuracy_rate=0.30)

# Scenario B: The AI is writing standard prose (Accuracy = 60%)
optimizer.find_optimal_shape(current_accuracy_rate=0.60)

# Scenario C: The AI is writing highly predictable code/text (Accuracy = 90%)
optimizer.find_optimal_shape(current_accuracy_rate=0.90)