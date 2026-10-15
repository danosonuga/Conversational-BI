import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from orchestrator import process_query

import os
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, "..", "data")
FIG_DIR = os.path.join(_BASE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

fact = pd.read_csv(f"{DATA_DIR}/Fact_Transactions.csv")
prod = pd.read_csv(f"{DATA_DIR}/Dim_Product.csv")
cust = pd.read_csv(f"{DATA_DIR}/Dim_Customer.csv")


def render_system_output(query, filename):
    result = process_query(query, fact, prod, cust)
    agg = result.get("_result_data")
    if agg is None:
        print(f"[{query}] -> {result['intent_classification']}: no chart (as expected)")
        return result

    plt.figure(figsize=(7.5, 4.5))
    agg.sort_values(ascending=False).plot(kind="bar", color="#2563eb", edgecolor="white")
    plt.title(f"System Output: \"{query}\"", fontsize=11)
    plt.ylabel(result["explainability_text"].split("|")[0].replace("Showing: ", ""))
    plt.xticks(rotation=20, ha="right")
    plt.figtext(0.5, -0.05, result["explainability_text"], wrap=True,
                ha="center", fontsize=9, style="italic", color="#444444")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{query}] -> saved {filename}")
    print(f"   Explainability tag: {result['explainability_text']}")
    return result


render_system_output("Show me total revenue by region for Q3", "system_output_revenue_by_region_q3.png")
render_system_output("Compare profit margin by category", "system_output_margin_by_category.png")

# Ambiguity path — show what the system produces instead of a chart
amb = process_query("Show me our best products", fact, prod, cust)
print()
print("Ambiguity path output (no chart rendered):")
print(amb)