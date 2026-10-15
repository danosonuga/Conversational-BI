import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, "..", "data")
OUT_DIR = os.path.join(_BASE, "..", "figures")
import os
os.makedirs(OUT_DIR, exist_ok=True)

fact = pd.read_csv(f"{DATA_DIR}/Fact_Transactions.csv", parse_dates=["Timestamp"])
cust = pd.read_csv(f"{DATA_DIR}/Dim_Customer.csv")
prod = pd.read_csv(f"{DATA_DIR}/Dim_Product.csv")

merged = fact.merge(cust, on="Customer_ID").merge(prod, on="Product_ID")
merged["Month"] = merged["Timestamp"].dt.month
merged["Quarter"] = merged["Timestamp"].dt.quarter

# --- 1. Overall summary ---------------------------------------------------
total_revenue = merged["Revenue"].sum()
total_txns = len(merged)
avg_order_value = merged["Revenue"].mean()

print("=== OVERALL SUMMARY ===")
print(f"Total transactions: {total_txns:,}")
print(f"Total revenue:      {total_revenue:,.2f}")
print(f"Avg order value:    {avg_order_value:,.2f}")
print()

# --- 2. Revenue by Region x Quarter (validates the Q3 Western dip) --------
pivot = merged.pivot_table(index="Region", columns="Quarter", values="Revenue", aggfunc="sum")
pivot["Q3_vs_Q2_pct_change"] = (pivot[3] - pivot[2]) / pivot[2] * 100
print("=== REVENUE BY REGION x QUARTER ===")
print(pivot.round(2))
print()
pivot.to_csv(f"{DATA_DIR}/summary_region_quarter.csv")

# --- 3. Monthly revenue trend (all regions) -------------------------------
monthly = merged.groupby("Month")["Revenue"].sum()
plt.figure(figsize=(8, 4.5))
monthly.plot(kind="line", marker="o", color="#2563eb")
plt.title("Total Monthly Revenue - All Regions (2025)")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_monthly_revenue_all_regions.png", dpi=150)
plt.close()

# --- 4. Monthly revenue trend, Western vs. other regions ------------------
west = merged[merged["Region"] == "Western"].groupby("Month")["Revenue"].sum()
other = merged[merged["Region"] != "Western"].groupby("Month")["Revenue"].sum() / 4  # avg per region
plt.figure(figsize=(8, 4.5))
plt.plot(west.index, west.values, marker="o", label="Western Region", color="#dc2626")
plt.plot(other.index, other.values, marker="s", label="Other Regions (avg)", color="#2563eb")
plt.axvspan(7, 9, color="grey", alpha=0.15, label="Q3 (Jul-Sep)")
plt.title("Monthly Revenue: Western Region vs. Other Regions (2025)")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_western_vs_other_regions.png", dpi=150)
plt.close()

# --- 5. Revenue by category (for Bar Chart heuristic demo) ----------------
cat_rev = merged.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 4.5))
cat_rev.plot(kind="bar", color="#059669")
plt.title("Total Revenue by Product Category (2025)")
plt.ylabel("Revenue")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_revenue_by_category.png", dpi=150)
plt.close()

print("Charts saved to", OUT_DIR)
print(os.listdir(OUT_DIR))