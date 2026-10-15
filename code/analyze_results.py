"""
analyze_results.py

Turns the filled-in participant_data_template.xlsx into the statistics and
charts required for Chapter 4, Section 4.4 (Usability Evaluation Results)
and Chapter 5 (Discussion).

USAGE:
    1. Fill in participant_data_template.xlsx with real session data
       (delete the yellow example row first).
    2. Run: python3 analyze_results.py participant_data_template.xlsx
    3. Read the printed summary + open /figures for charts.

This script performs NO fabrication — it only computes statistics from
whatever rows are actually present in the spreadsheet. If the file still
contains only the example row, the output is clearly labeled as such.
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUS_ODD = [1, 3, 5, 7, 9]     # 1-indexed SUS item numbers scored as (response - 1)
SUS_EVEN = [2, 4, 6, 8, 10]   # scored as (5 - response)

def sus_score(row):
    total = 0
    for i in SUS_ODD:
        total += row[f"SUS_Q{i}"] - 1
    for i in SUS_EVEN:
        total += 5 - row[f"SUS_Q{i}"]
    return total * 2.5

def sus_adjective(score):
    # Bangor, Kortum & Miller (2009) adjective rating scale
    if score >= 85: return "Best Imaginable"
    if score >= 72.6: return "Excellent"
    if score >= 52: return "Good"
    if score >= 39: return "OK"
    if score >= 25: return "Poor"
    return "Worst Imaginable"

def main(path):
    df = pd.read_excel(path, sheet_name="Participant Data")

    is_example = (df["Participant_ID"] == "P01").any() and len(df) == 1
    if is_example:
        print("WARNING: Only the example row is present. Delete it and add")
        print("real participant rows before drawing conclusions.\n")

    df = df[df["Participant_ID"] != "P01"].copy() if len(df) > 1 else df

    if df.empty:
        print("No participant data found. Nothing to analyze yet.")
        return

    df["SUS_Score"] = df.apply(sus_score, axis=1)
    df["SUS_Adjective"] = df["SUS_Score"].apply(sus_adjective)

    print("=== SUS RESULTS ===")
    print(f"N = {len(df)}")
    print(f"Mean SUS score: {df['SUS_Score'].mean():.1f}")
    print(f"Median SUS score: {df['SUS_Score'].median():.1f}")
    print(f"Std dev: {df['SUS_Score'].std():.1f}")
    print(df[["Participant_ID", "SUS_Score", "SUS_Adjective"]].to_string(index=False))
    print()

    print("=== TASK COMPLETION RATES ===")
    for t in ["T1", "T2", "T3", "T4"]:
        col = f"{t}_Completed_YN"
        rate = (df[col].str.lower() == "yes").mean() * 100
        print(f"{t}: {rate:.1f}% completed")
    print()

    print("=== INSIGHT LATENCY (seconds, mean) ===")
    for t in ["T1", "T2", "T3", "T4"]:
        col = f"{t}_Latency_Seconds"
        print(f"{t}: {df[col].mean():.1f}s (median {df[col].median():.1f}s)")
    print()

    print("=== TRUST ITEMS (mean, 1-5 scale) ===")
    for col, label in [
        ("Trust_Understood_Logic_1to5", "Understood AI logic"),
        ("Trust_Would_Trust_Decision_1to5", "Would trust a decision from output"),
        ("Trust_Clarifications_Helpful_1to5", "Clarifications felt helpful"),
    ]:
        print(f"{label}: {df[col].mean():.2f}")
    print()

    # Charts
    import os
    FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    plt.figure(figsize=(7, 4))
    df["SUS_Score"].hist(bins=10, color="#2563eb", edgecolor="white")
    plt.axvline(df["SUS_Score"].mean(), color="red", linestyle="--", label="Mean")
    plt.title("Distribution of SUS Scores")
    plt.xlabel("SUS Score")
    plt.ylabel("Participants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_sus_distribution.png"), dpi=150)
    plt.close()

    completion = [(df[f"{t}_Completed_YN"].str.lower() == "yes").mean() * 100 for t in ["T1","T2","T3","T4"]]
    plt.figure(figsize=(7, 4))
    plt.bar(["T1 Happy Path", "T2 Ambiguity", "T3 Multi-turn", "T4 Explainability"], completion, color="#059669")
    plt.ylabel("Completion Rate (%)")
    plt.title("Task Completion Rate by Scenario")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_task_completion.png"), dpi=150)
    plt.close()

    print(f"Charts saved to {FIG_DIR}/")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "participant_data_template.xlsx"
    main(path)