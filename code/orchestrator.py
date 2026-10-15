"""
orchestrator.py

The Orchestration / Semantic Context / Ambiguity Resolution logic described
in Chapter 3 (Sections 3.3.2-3.3.4) implemented as deterministic, rule-based
Python code, rather than left purely as instructions for a human "wizard"
to follow. This lets the interaction logic itself be unit- and
integration-tested (Chapter 5), producing real, reproducible performance
and correctness metrics rather than only human-facilitated demonstrations.

Scope note (see Chapter 4, Section 4.2): this module intentionally uses
keyword/pattern matching rather than a trained LLM. It exists to give the
WoZ study a consistent, auditable "brain" that a facilitator can also
follow manually during live sessions, and to allow automated regression
testing of the interaction rules independent of a live LLM API (which is
explicitly out of scope per Section 1.4).
"""

import re
import time
import pandas as pd

# ---------------------------------------------------------------------------
# Semantic Context Layer: canonical metric/dimension definitions
# ---------------------------------------------------------------------------
METRIC_SYNONYMS = {
    "revenue": ["revenue", "sales", "income", "turnover"],
    "margin": ["margin", "profit margin", "profitability"],
}
AMBIGUOUS_METRIC_WORDS = ["best", "worst", "top", "highest performing", "performing"]
DIMENSION_KEYWORDS = {
    "region": ["region", "regions", "regional"],
    "category": ["category", "categories", "product type"],
}
REGIONS = ["Western", "Eastern", "Northern", "Southern", "Central"]
QUARTER_PATTERN = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)


def extract_entities(query: str) -> dict:
    """Natural Language Understanding (NLU) stage: tokenises the raw
    utterance and extracts Metric, Timeframe, Dimension, and Region
    entities per Section 3.3.2."""
    q = query.lower()

    metric = None
    for canonical, synonyms in METRIC_SYNONYMS.items():
        if any(s in q for s in synonyms):
            metric = canonical
            break

    ambiguous_metric = any(w in q for w in AMBIGUOUS_METRIC_WORDS) and metric is None

    dimension = None
    for canonical, keywords in DIMENSION_KEYWORDS.items():
        if any(k in q for k in keywords):
            dimension = canonical
            break

    quarter_match = QUARTER_PATTERN.search(query)
    timeframe = f"Q{quarter_match.group(1)}" if quarter_match else None
    if "this year" in q or "2025" in q:
        timeframe = timeframe or "FY2025"

    region = next((r for r in REGIONS if r.lower() in q), None)

    return {
        "metric": metric,
        "ambiguous_metric": ambiguous_metric,
        "dimension": dimension,
        "timeframe": timeframe,
        "region": region,
    }


def classify_intent(entities: dict) -> str:
    """Zero-shot intent classification per Section 3.3.2."""
    if entities["ambiguous_metric"]:
        return "Clarification_Required"
    if entities["dimension"]:
        return "Data_Comparison"
    if entities["metric"]:
        return "Data_Retrieval"
    return "Schema_Exploration"


AMBIGUITY_CLARIFICATIONS = {
    "best": ["Total Revenue", "Highest Profit Margin", "Total Units Sold"],
    "worst": ["Lowest Revenue", "Lowest Profit Margin", "Fewest Units Sold"],
    "top": ["Total Revenue", "Highest Profit Margin", "Total Units Sold"],
}


def build_clarification_question(query: str) -> str:
    q = query.lower()
    for trigger, options in AMBIGUITY_CLARIFICATIONS.items():
        if trigger in q:
            opts = ", ".join(f"[{chr(65+i)}] {o}" for i, o in enumerate(options))
            return f"I can help with that. Would you like me to sort by: {opts}?"
    return "Could you clarify which metric you'd like me to use?"


class OutOfSchemaError(Exception):
    """Raised when a query maps to a metric not present in the simulated
    semantic layer (Section 3.4.1) -- triggers graceful degradation (NFR3)."""
    pass


def map_semantic_layer(entities: dict, fact_df: pd.DataFrame, product_df: pd.DataFrame,
                        customer_df: pd.DataFrame) -> pd.DataFrame:
    """Maps validated entities onto the simulated Fact_Transactions /
    Dim_Product / Dim_Customer schema (Section 3.4.1) and returns the
    filtered, aggregation-ready dataframe. Raises OutOfSchemaError for
    metrics that do not exist in the schema (e.g. 'units sold' -- the
    schema defined in Ch.3 does not persist a quantity column),
    demonstrating NFR3 (graceful degradation)."""
    if entities["metric"] is None:
        raise OutOfSchemaError("No resolvable metric in the simulated schema.")

    df = fact_df.merge(customer_df, on="Customer_ID").merge(product_df, on="Product_ID")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Quarter"] = "Q" + df["Timestamp"].dt.quarter.astype(str)

    if entities["timeframe"] and entities["timeframe"].startswith("Q"):
        df = df[df["Quarter"] == entities["timeframe"]]
    if entities["region"]:
        df = df[df["Region"] == entities["region"]]

    return df


def generate_json_envelope(query: str, entities: dict, intent: str,
                            chart_type: str = None, explainability_text: str = None,
                            clarification_question: str = None) -> dict:
    """Produces the standardized JSON output envelope defined in
    Section 3.4.5."""
    return {
        "intent_classification": intent,
        "extracted_entities": [e for e in
                                [entities.get("metric"), entities.get("timeframe"),
                                 entities.get("dimension"), entities.get("region")] if e],
        "ambiguity_detected": intent == "Clarification_Required",
        "clarification_question": clarification_question,
        "chart_type": chart_type,
        "explainability_text": explainability_text,
    }


def select_chart_type(dimension: str) -> str:
    """Heuristic Chart Selection per Section 3.3.5."""
    if dimension in ("region", "category"):
        return "bar"
    return "line"


def process_query(query: str, fact_df: pd.DataFrame, product_df: pd.DataFrame,
                   customer_df: pd.DataFrame) -> dict:
    """Full end-to-end pipeline: NLU -> Intent Classification -> Semantic
    Mapping -> Ambiguity Check -> Output Envelope. Returns the JSON
    envelope plus (if resolvable) the aggregated result data."""
    entities = extract_entities(query)
    intent = classify_intent(entities)

    if intent == "Clarification_Required":
        cq = build_clarification_question(query)
        return generate_json_envelope(query, entities, intent, clarification_question=cq)

    try:
        df = map_semantic_layer(entities, fact_df, product_df, customer_df)
    except OutOfSchemaError:
        return generate_json_envelope(
            query, entities, "Out_Of_Scope",
            explainability_text="This metric is not available in the connected schema. "
                                 "Try: revenue or margin, by region or category."
        )

    if entities["metric"] == "revenue":
        group_col = "Region" if entities["dimension"] == "region" else (
            "Category" if entities["dimension"] == "category" else None)
        agg = df.groupby(group_col)["Revenue"].sum() if group_col else df["Revenue"].sum()
        metric_label = "Total Revenue"
    else:  # margin
        group_col = "Region" if entities["dimension"] == "region" else (
            "Category" if entities["dimension"] == "category" else None)
        agg = df.groupby(group_col)["Margin"].mean() if group_col else df["Margin"].mean()
        metric_label = "Average Profit Margin"

    chart_type = select_chart_type(entities["dimension"])
    filters = []
    if entities["timeframe"]:
        filters.append(f"Timeframe={entities['timeframe']}")
    if entities["region"]:
        filters.append(f"Region={entities['region']}")
    explain = f"Showing: {metric_label} | Grouped by: {entities['dimension'] or 'none'} | Filters: {', '.join(filters) or 'none'}"

    envelope = generate_json_envelope(query, entities, intent, chart_type=chart_type, explainability_text=explain)
    envelope["_result_data"] = agg  # not part of the formal schema; used internally for rendering
    return envelope


def timed_process_query(query, fact_df, product_df, customer_df):
    """Wraps process_query with latency measurement for performance testing
    (Chapter 5, Section 5.2)."""
    start = time.perf_counter()
    result = process_query(query, fact_df, product_df, customer_df)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms