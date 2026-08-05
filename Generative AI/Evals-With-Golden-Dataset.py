import pandas as pd
import numpy as np

# 1. Define the Ground Truth Golden Dataset
golden_data = {
    "case_id": ["TC_001", "TC_002", "TC_003"],
    "user_query": [
        "What is the cancellation policy for Enterprise Tier?",
        "How do I update my billing credit card?",
        "Can I download financial data via API?"
    ],
    "expected_context": [
        "Enterprise contracts require a 30-day written cancellation notice.",
        "Navigate to Settings -> Billing -> Payment Methods to update cards.",
        "Yes, API endpoints for financial data extraction are fully accessible."
    ]
}
df_golden = pd.DataFrame(golden_data)
df_golden

# 2. Simulate the Live API Execution Snippet (Actual Model Outputs)
df_golden["actual_output"] = [
    "Enterprise clients must provide a 30-day notice in writing to cancel.",
    "Go to the account profile tab and look for the billing options link.",
    "We do not support financial data downloads via the web interface." # Hallucination/Error
]

# 3. Apply the Token Calculation Formula and Metric Evaluation
df_golden["semantic_precision"] = [0.95, 0.82, 0.10]
df_golden["semantic_recall"] = [0.98, 0.85, 0.05]
df_golden["latency_ms"] = [450, 620, 1100]
df_golden["tokens_used"] = [120, 95, 340]

# Calculate Cost ($0.0018 per 1k tokens combined average)
cost_per_token = 0.0018 / 1000
df_golden["execution_cost"] = df_golden["tokens_used"] * cost_per_token

# 4. Display the Confusion Matrix Evaluation Table
print("=== ENTERPRISE LLM BENCHMARK RESULT ===")
print(df_golden[["case_id", "semantic_precision", "latency_ms", "execution_cost"]])

# Compute systemic F1 aggregate approximation
avg_precision = df_golden["semantic_precision"].mean()
avg_recall = df_golden["semantic_recall"].mean()
f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

print(f"\n🚀 System Baseline F1-Score: {f1_score:.2f}")
print(f"💰 Mean Cost per Task: ${df_golden['execution_cost'].mean():.6f}")
