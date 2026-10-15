"""
generate_dataset.py

Generates the simulated Enterprise E-Commerce dataset described in
Chapter 3, Section 3.4.1 (Simulated Database Design / Semantic Layer Schema)
of the project:

    "Evaluating Human-AI Interaction Paradigms in Intent-Driven
    Conversational Dashboards for Business Intelligence"

Three tables are produced, matching the schema defined in the thesis:
    1. Dim_Customer       (Customer_ID, Region, Acquisition_Date, Subscription_Status)
    2. Dim_Product        (Product_ID, Category, Unit_Cost, Margin)
    3. Fact_Transactions   (Transaction_ID, Customer_ID, Product_ID, Timestamp,
                            Revenue, Discount_Applied)

Deliberate statistical properties are injected so that the Wizard-of-Oz
operator can produce believable, mathematically consistent charts during
usability testing:
    - Seasonal / monthly variance in transaction volume
    - A deliberate Q3 (Jul-Sep) revenue dip in the "Western" region, so the
      "Why did revenue drop in Q3?" scenario used in the WoZ script has a
      real, discoverable answer in the underlying data
    - Realistic margin/discount structure by product category

No real, proprietary, or personally identifiable data is used anywhere in
this script, consistent with the ethical scope boundary set in Chapter 1
and Chapter 3 (Section 3.2.3).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

import os
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Dim_Customer
# ---------------------------------------------------------------------------
N_CUSTOMERS = 2500
REGIONS = ["Western", "Eastern", "Northern", "Southern", "Central"]
REGION_WEIGHTS = [0.30, 0.22, 0.16, 0.18, 0.14]  # Western is the largest region

customer_ids = np.arange(1, N_CUSTOMERS + 1)
regions = rng.choice(REGIONS, size=N_CUSTOMERS, p=REGION_WEIGHTS)

# Acquisition dates spread across the last 3 years
start_acq = datetime(2023, 1, 1)
acq_days = rng.integers(0, 365 * 3, size=N_CUSTOMERS)
acquisition_dates = [start_acq + timedelta(days=int(d)) for d in acq_days]

subscription_status = rng.choice(
    ["Active", "Inactive", "Trial"], size=N_CUSTOMERS, p=[0.62, 0.28, 0.10]
)

dim_customer = pd.DataFrame({
    "Customer_ID": customer_ids,
    "Region": regions,
    "Acquisition_Date": [d.strftime("%Y-%m-%d") for d in acquisition_dates],
    "Subscription_Status": subscription_status,
})

# ---------------------------------------------------------------------------
# 2. Dim_Product
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Electronics":      {"cost_range": (40, 400),  "margin_range": (0.10, 0.22)},
    "Home & Living":    {"cost_range": (10, 150),  "margin_range": (0.20, 0.35)},
    "Fashion":          {"cost_range": (5, 90),     "margin_range": (0.30, 0.55)},
    "Groceries":        {"cost_range": (2, 40),     "margin_range": (0.05, 0.15)},
    "Beauty & Personal Care": {"cost_range": (3, 60), "margin_range": (0.25, 0.45)},
}

N_PRODUCTS = 400
product_ids = np.arange(1, N_PRODUCTS + 1)
product_categories = rng.choice(list(CATEGORIES.keys()), size=N_PRODUCTS)

unit_costs = []
margins = []
for cat in product_categories:
    cost_lo, cost_hi = CATEGORIES[cat]["cost_range"]
    m_lo, m_hi = CATEGORIES[cat]["margin_range"]
    unit_costs.append(round(rng.uniform(cost_lo, cost_hi), 2))
    margins.append(round(rng.uniform(m_lo, m_hi), 3))

dim_product = pd.DataFrame({
    "Product_ID": product_ids,
    "Category": product_categories,
    "Unit_Cost": unit_costs,
    "Margin": margins,
})

# ---------------------------------------------------------------------------
# 3. Fact_Transactions
# ---------------------------------------------------------------------------
N_TRANSACTIONS = 45000
DATE_START = datetime(2025, 1, 1)
DATE_END = datetime(2025, 12, 31)
total_days = (DATE_END - DATE_START).days

# Base daily seasonality: slightly higher volume Nov-Dec (holiday season),
# a dip in Jul-Sep for the Western region specifically (the Q3 anomaly),
# gentle upward trend across the year overall.
def day_weight(day_offset):
    date = DATE_START + timedelta(days=int(day_offset))
    month = date.month
    weight = 1.0
    if month in (11, 12):
        weight *= 1.35          # holiday boost
    if month in (1, 2):
        weight *= 0.85          # post-holiday lull
    weight *= 1 + (day_offset / total_days) * 0.15  # mild yearly growth trend
    return weight

day_offsets = np.arange(total_days + 1)
day_weights = np.array([day_weight(d) for d in day_offsets])
day_probs = day_weights / day_weights.sum()

txn_day_offsets = rng.choice(day_offsets, size=N_TRANSACTIONS, p=day_probs)
txn_dates = [DATE_START + timedelta(days=int(d)) for d in txn_day_offsets]
txn_customers = rng.choice(customer_ids, size=N_TRANSACTIONS)

# Map each transaction's customer to a region so we can apply the Q3 Western dip
cust_region_map = dict(zip(dim_customer["Customer_ID"], dim_customer["Region"]))
txn_regions = np.array([cust_region_map[c] for c in txn_customers])

txn_products = rng.choice(product_ids, size=N_TRANSACTIONS)
prod_cost_map = dict(zip(dim_product["Product_ID"], dim_product["Unit_Cost"]))
prod_margin_map = dict(zip(dim_product["Product_ID"], dim_product["Margin"]))

quantities = rng.integers(1, 5, size=N_TRANSACTIONS)
unit_prices = np.array([
    prod_cost_map[p] * (1 + prod_margin_map[p]) for p in txn_products
])

discount_applied = rng.choice(
    [0, 0.05, 0.10, 0.15, 0.20], size=N_TRANSACTIONS, p=[0.55, 0.20, 0.15, 0.07, 0.03]
)

gross_revenue = unit_prices * quantities
revenue = gross_revenue * (1 - discount_applied)

# --- Inject the deliberate Q3 "Western region" revenue dip ---------------
months = np.array([d.month for d in txn_dates])
is_q3 = np.isin(months, [7, 8, 9])
is_western = txn_regions == "Western"
q3_western_mask = is_q3 & is_western

# Cut revenue by ~34% for Western-region transactions falling in Q3, and
# additionally drop a fraction of those transactions to simulate reduced
# purchase frequency (not just discounting) — mirrors a churn/awareness event.
revenue[q3_western_mask] *= 0.66
keep_mask = np.ones(N_TRANSACTIONS, dtype=bool)
drop_indices = np.where(q3_western_mask)[0]
n_drop = int(len(drop_indices) * 0.30)
if n_drop > 0:
    drop_choice = rng.choice(drop_indices, size=n_drop, replace=False)
    keep_mask[drop_choice] = False

fact_transactions = pd.DataFrame({
    "Transaction_ID": np.arange(1, N_TRANSACTIONS + 1),
    "Customer_ID": txn_customers,
    "Product_ID": txn_products,
    "Timestamp": [d.strftime("%Y-%m-%d") for d in txn_dates],
    "Revenue": np.round(revenue, 2),
    "Discount_Applied": discount_applied,
})[keep_mask].reset_index(drop=True)
fact_transactions["Transaction_ID"] = np.arange(1, len(fact_transactions) + 1)

# ---------------------------------------------------------------------------
# Persist to CSV
# ---------------------------------------------------------------------------
dim_customer.to_csv(f"{OUTPUT_DIR}/Dim_Customer.csv", index=False)
dim_product.to_csv(f"{OUTPUT_DIR}/Dim_Product.csv", index=False)
fact_transactions.to_csv(f"{OUTPUT_DIR}/Fact_Transactions.csv", index=False)

print("Dataset generated successfully.")
print(f"  Dim_Customer:      {len(dim_customer):,} rows")
print(f"  Dim_Product:       {len(dim_product):,} rows")
print(f"  Fact_Transactions: {len(fact_transactions):,} rows")