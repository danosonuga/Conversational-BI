"""
integration_test.py

Runs a batch of realistic natural-language queries through the full
orchestrator pipeline end-to-end (NLU -> Intent Classification ->
Semantic Mapping -> Rendering) and records:
    - per-query latency (ms)
    - correctness against a manually labeled expected outcome
    - whether paraphrased/colloquial variants of the same underlying
      question are handled consistently (a known brittleness risk of
      keyword-based NLU, discussed in Chapter 4.4 / Chapter 5.4)

This produces the real performance and correctness figures reported in
Chapter 5, Section 5.2 -- nothing here is hand-typed after the fact; the
script below is what generated them.
"""

import pandas as pd
from orchestrator import timed_process_query

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

fact = pd.read_csv(f"{DATA_DIR}/Fact_Transactions.csv")
prod = pd.read_csv(f"{DATA_DIR}/Dim_Product.csv")
cust = pd.read_csv(f"{DATA_DIR}/Dim_Customer.csv")

# label = expected intent_classification
TEST_QUERIES = [
    ("Show me total revenue by region for Q3", "Data_Comparison"),
    ("What was our total revenue in Q3?", "Data_Retrieval"),
    ("Show me our best products", "Clarification_Required"),
    ("Show me our worst performing assets", "Clarification_Required"),
    ("Compare profit margin by category", "Data_Comparison"),
    ("Show me employee payroll costs", "Out_Of_Scope"),
    ("How did the Western region do in Q2?", "Data_Retrieval"),
    # --- paraphrased / colloquial variants of the SAME underlying happy-path question ---
    ("Q3 revenue please, broken down region by region", "Data_Comparison"),   # reordered phrasing
    ("gimme the sales numbers for q3 across regions", "Data_Comparison"),      # colloquial synonym "sales", lowercase q3
    ("Can you break revenue down regionally for the third quarter?", "Data_Comparison"),  # "third quarter" instead of Q3
]

results = []
for query, expected_intent in TEST_QUERIES:
    envelope, latency_ms = timed_process_query(query, fact, prod, cust)
    correct = envelope["intent_classification"] == expected_intent
    results.append({
        "query": query,
        "expected": expected_intent,
        "actual": envelope["intent_classification"],
        "correct": correct,
        "latency_ms": round(latency_ms, 3),
    })

df = pd.DataFrame(results)
pd.set_option("display.max_colwidth", 60)
print(df.to_string(index=False))
print()
print(f"Overall accuracy: {df['correct'].mean()*100:.1f}% ({df['correct'].sum()}/{len(df)})")
print(f"Mean latency: {df['latency_ms'].mean():.3f} ms")
print(f"Max latency: {df['latency_ms'].max():.3f} ms")

df.to_csv(os.path.join(DATA_DIR, "integration_test_results.csv"), index=False)