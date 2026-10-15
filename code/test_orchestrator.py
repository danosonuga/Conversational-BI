import unittest
import pandas as pd
from orchestrator import (
    extract_entities, classify_intent, build_clarification_question,
    map_semantic_layer, OutOfSchemaError, process_query, select_chart_type
)

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


class TestEntityExtraction(unittest.TestCase):
    def test_happy_path_query(self):
        e = extract_entities("Show me total revenue by region for Q3")
        self.assertEqual(e["metric"], "revenue")
        self.assertEqual(e["dimension"], "region")
        self.assertEqual(e["timeframe"], "Q3")
        self.assertFalse(e["ambiguous_metric"])

    def test_ambiguous_query(self):
        e = extract_entities("Show me our best products")
        self.assertIsNone(e["metric"])
        self.assertTrue(e["ambiguous_metric"])

    def test_margin_synonym_recognized(self):
        e = extract_entities("What is our profit margin by category?")
        self.assertEqual(e["metric"], "margin")
        self.assertEqual(e["dimension"], "category")

    def test_region_entity_recognized(self):
        e = extract_entities("How did the Western region perform in Q2?")
        self.assertEqual(e["region"], "Western")
        self.assertEqual(e["timeframe"], "Q2")

    def test_out_of_schema_query_has_no_metric(self):
        e = extract_entities("Show me employee payroll costs")
        self.assertIsNone(e["metric"])
        self.assertFalse(e["ambiguous_metric"])

    def test_case_insensitivity(self):
        e1 = extract_entities("REVENUE BY REGION Q1")
        self.assertEqual(e1["metric"], "revenue")
        self.assertEqual(e1["timeframe"], "Q1")


class TestIntentClassification(unittest.TestCase):
    def test_ambiguous_routes_to_clarification(self):
        e = extract_entities("Show our worst performing assets")
        self.assertEqual(classify_intent(e), "Clarification_Required")

    def test_dimension_present_routes_to_comparison(self):
        e = extract_entities("Compare revenue by category")
        self.assertEqual(classify_intent(e), "Data_Comparison")

    def test_metric_only_routes_to_retrieval(self):
        e = extract_entities("What was total revenue in Q3?")
        self.assertEqual(classify_intent(e), "Data_Retrieval")

    def test_no_entities_routes_to_schema_exploration(self):
        e = extract_entities("What can you tell me about?")
        self.assertEqual(classify_intent(e), "Schema_Exploration")


class TestClarificationGeneration(unittest.TestCase):
    def test_best_triggers_three_options(self):
        q = build_clarification_question("show me our best products")
        self.assertIn("Total Revenue", q)
        self.assertIn("Highest Profit Margin", q)
        self.assertIn("Total Units Sold", q)


class TestChartSelectionHeuristic(unittest.TestCase):
    def test_region_dimension_selects_bar(self):
        self.assertEqual(select_chart_type("region"), "bar")

    def test_no_dimension_selects_line(self):
        self.assertEqual(select_chart_type(None), "line")


class TestSemanticMappingAndOutOfSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fact = pd.read_csv(f"{DATA_DIR}/Fact_Transactions.csv")
        cls.prod = pd.read_csv(f"{DATA_DIR}/Dim_Product.csv")
        cls.cust = pd.read_csv(f"{DATA_DIR}/Dim_Customer.csv")

    def test_out_of_schema_raises(self):
        entities = {"metric": None, "ambiguous_metric": False, "dimension": None,
                    "timeframe": None, "region": None}
        with self.assertRaises(OutOfSchemaError):
            map_semantic_layer(entities, self.fact, self.prod, self.cust)

    def test_valid_query_returns_filtered_frame(self):
        entities = {"metric": "revenue", "ambiguous_metric": False, "dimension": "region",
                    "timeframe": "Q3", "region": None}
        df = map_semantic_layer(entities, self.fact, self.prod, self.cust)
        self.assertTrue((df["Quarter"] == "Q3").all())
        self.assertGreater(len(df), 0)


class TestEndToEndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fact = pd.read_csv(f"{DATA_DIR}/Fact_Transactions.csv")
        cls.prod = pd.read_csv(f"{DATA_DIR}/Dim_Product.csv")
        cls.cust = pd.read_csv(f"{DATA_DIR}/Dim_Customer.csv")

    def test_happy_path_end_to_end(self):
        result = process_query("Show me total revenue by region for Q3",
                                self.fact, self.prod, self.cust)
        self.assertEqual(result["intent_classification"], "Data_Comparison")
        self.assertFalse(result["ambiguity_detected"])
        self.assertEqual(result["chart_type"], "bar")
        self.assertIn("_result_data", result)
        # The known injected anomaly is a relative Q3 drop for Western, not an
        # absolute minimum (Western is the largest region overall, so its Q3
        # figure can still exceed smaller regions' Q3 figures in absolute terms)
        agg = result["_result_data"]
        result_q2 = process_query("Show me total revenue by region for Q2",
                                   self.fact, self.prod, self.cust)
        agg_q2 = result_q2["_result_data"]
        western_drop_pct = (agg["Western"] - agg_q2["Western"]) / agg_q2["Western"] * 100
        self.assertLess(western_drop_pct, -30)  # Western fell more than 30% Q2->Q3
        other_regions = [r for r in agg.index if r != "Western"]
        other_avg_change_pct = sum(
            (agg[r] - agg_q2[r]) / agg_q2[r] * 100 for r in other_regions
        ) / len(other_regions)
        self.assertGreater(other_avg_change_pct, 0)  # other regions grew Q2->Q3

    def test_ambiguity_path_end_to_end(self):
        result = process_query("Show me our best products", self.fact, self.prod, self.cust)
        self.assertTrue(result["ambiguity_detected"])
        self.assertIsNotNone(result["clarification_question"])
        self.assertNotIn("_result_data", result)

    def test_out_of_schema_end_to_end(self):
        result = process_query("Show me employee payroll costs", self.fact, self.prod, self.cust)
        self.assertEqual(result["intent_classification"], "Out_Of_Scope")
        self.assertIsNotNone(result["explainability_text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)