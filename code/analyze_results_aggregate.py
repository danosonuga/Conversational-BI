"""
analyze_results_aggregate.py

A variant of analyze_results.py for the case where only Maze's AGGREGATE
report (percentages/counts per question) is available, rather than a
row-per-participant export. This is exactly the situation for the
"Conversational BI" Maze study (15 responses, 8 completing the SUS block).

The SUS_DISTRIBUTIONS and TASK_METRICS dictionaries below are transcribed
directly from the Maze PDF report (Coversation_BI_usability.pdf) supplied
by the user. Nothing here is invented or estimated -- every number traces
to a specific value shown in that report.

Mathematical note: because every SUS item in the report has the same N (8),
computing the mean item score from each item's distribution and summing is
exactly equivalent to averaging 8 individual SUS totals -- this is not an
approximation.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- Transcribed directly from the Maze report (rating: count of respondents) ---
SUS_DISTRIBUTIONS = {
    1:  {"text": "I think that I would like to use this system frequently.",
         "odd": True,  "dist": {3: 3, 4: 2, 5: 2, 2: 1, 1: 0}},
    2:  {"text": "I found the system unnecessarily complex.",
         "odd": False, "dist": {3: 3, 1: 2, 4: 2, 2: 1, 5: 0}},
    3:  {"text": "I thought the system was easy to use.",
         "odd": True,  "dist": {3: 3, 2: 2, 1: 1, 4: 1, 5: 1}},
    4:  {"text": "I think that I would need the support of a technical person.",
         "odd": False, "dist": {2: 4, 1: 3, 4: 1, 3: 0, 5: 0}},
    5:  {"text": "I found the various functions in this system were well integrated.",
         "odd": True,  "dist": {5: 4, 3: 2, 4: 2, 1: 0, 2: 0}},
    6:  {"text": "I thought there was too much inconsistency in this system.",
         "odd": False, "dist": {2: 6, 1: 1, 4: 1, 3: 0, 5: 0}},
    7:  {"text": "I would imagine that most people would learn to use this system very quickly.",
         "odd": True,  "dist": {4: 4, 3: 3, 2: 1, 1: 0, 5: 0}},
    8:  {"text": "I found the system very cumbersome to use.",
         "odd": False, "dist": {2: 3, 3: 2, 1: 1, 4: 1, 5: 1}},
    9:  {"text": "I felt very confident using the system.",
         "odd": True,  "dist": {3: 3, 5: 2, 1: 1, 2: 1, 4: 1}},
    10: {"text": "I needed to learn a lot of things before I could get going with this system.",
         "odd": False, "dist": {1: 4, 2: 2, 4: 1, 5: 1, 3: 0}},
}

TASK_METRICS = {
    "T1 Happy Path":        {"responses": 11, "success_rate": 72.7, "expected_path": 45.5, "misclick": 79.8, "duration_s": 118.8},
    "T2 Ambiguity":         {"responses": 10, "success_rate": 90.0, "expected_path": 70.0, "misclick": 66.7, "duration_s": 34.0},
    "T3 Multi-turn":        {"responses": 10, "success_rate": 100.0, "expected_path": None, "misclick": 23.1, "duration_s": 8.3},
    "T4 Explainability":    {"responses": 8,  "success_rate": 87.5, "expected_path": None, "misclick": 89.2, "duration_s": 37.3},
}


def compute_sus():
    item_scores = {}
    for q, info in SUS_DISTRIBUTIONS.items():
        dist = info["dist"]
        n = sum(dist.values())
        mean_rating = sum(r * c for r, c in dist.items()) / n
        score = (mean_rating - 1) if info["odd"] else (5 - mean_rating)
        item_scores[q] = {"mean_rating": mean_rating, "score": score, "n": n, "text": info["text"]}
    total = sum(v["score"] for v in item_scores.values())
    sus = total * 2.5
    return item_scores, sus


def adjective(score):
    if score >= 85: return "Best Imaginable"
    if score >= 72.6: return "Excellent"
    if score >= 52: return "Good"
    if score >= 39: return "OK"
    if score >= 25: return "Poor"
    return "Worst Imaginable"


def main():
    item_scores, sus = compute_sus()

    print("=== SUS RESULTS (from Maze aggregate distributions, N=8) ===")
    for q in range(1, 11):
        v = item_scores[q]
        print(f"Q{q}: mean rating={v['mean_rating']:.3f}  item score={v['score']:.3f}  (\"{v['text'][:50]}...\")")
    print(f"\nSum of item scores: {sum(v['score'] for v in item_scores.values()):.3f}")
    print(f"SUS score: {sus:.1f}")
    print(f"Adjective rating: {adjective(sus)}")
    print(f"Vs. industry benchmark (68): {'above' if sus > 68 else 'below'} by {abs(sus-68):.1f} points")
    print()

    print("=== TASK METRICS (from Maze report) ===")
    for task, m in TASK_METRICS.items():
        print(f"{task}: N={m['responses']}, success={m['success_rate']}%, "
              f"expected_path={m['expected_path']}%, misclick={m['misclick']}%, "
              f"avg_duration={m['duration_s']}s")

    # --- Chart 1: per-item SUS scores (bar) ---
    plt.figure(figsize=(9, 4.5))
    qs = list(range(1, 11))
    scores = [item_scores[q]["score"] for q in qs]
    colors = ["#2563eb" if SUS_DISTRIBUTIONS[q]["odd"] else "#059669" for q in qs]
    plt.bar([f"Q{q}" for q in qs], scores, color=colors)
    plt.axhline(2.5, color="grey", linestyle="--", linewidth=1, label="Neutral midpoint (2.5)")
    plt.ylabel("SUS item score (0-4 scale)")
    plt.title(f"SUS Item Scores (N=8) — Overall SUS = {sus:.1f} \"{adjective(sus)}\"")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_sus_item_scores.png", dpi=150)
    plt.close()

    # --- Chart 2: task success rate vs expected-path completion ---
    plt.figure(figsize=(8, 4.5))
    tasks = list(TASK_METRICS.keys())
    success = [TASK_METRICS[t]["success_rate"] for t in tasks]
    expected = [TASK_METRICS[t]["expected_path"] if TASK_METRICS[t]["expected_path"] is not None else float("nan") for t in tasks]
    x = range(len(tasks))
    width = 0.35
    plt.bar([i - width/2 for i in x], success, width, label="Success rate", color="#2563eb")
    plt.bar([i + width/2 for i in x], expected, width, label="Completed via expected path", color="#dc2626")
    plt.xticks(list(x), tasks, rotation=15, ha="right")
    plt.ylabel("%")
    plt.ylim(0, 110)
    plt.title("Task Success Rate vs. Expected-Path Completion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_task_completion.png", dpi=150)
    plt.close()

    # --- Chart 3: average duration per task ---
    plt.figure(figsize=(7, 4.5))
    durations = [TASK_METRICS[t]["duration_s"] for t in tasks]
    plt.bar(tasks, durations, color="#f59e0b")
    plt.ylabel("Average duration (seconds)")
    plt.title("Average Task Duration")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_task_duration.png", dpi=150)
    plt.close()

    print(f"\nCharts saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()