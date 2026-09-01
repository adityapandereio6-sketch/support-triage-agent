import unittest
from unittest.mock import patch

from triage_agent import (
    KNOWLEDGE_BASE,
    SklearnRAGEngine,
    TriageAgent,
)


class TestSklearnRAGEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SklearnRAGEngine(KNOWLEDGE_BASE)

    def test_retrieve_returns_valid_result(self):
        index, context, similarity = self.engine.retrieve(
            "What did the ALPHA-g experiment show about antimatter?"
        )

        self.assertIsInstance(index, int)
        self.assertIsInstance(context, str)
        self.assertIsInstance(similarity, float)

        self.assertGreaterEqual(index, 0)
        self.assertLess(index, len(KNOWLEDGE_BASE))
        self.assertGreater(similarity, 0)


class TestTriageAgent(unittest.TestCase):

    def setUp(self):
        self.agent = TriageAgent(KNOWLEDGE_BASE)

    def test_risky_query_is_sent_to_human_triage(self):
        result = self.agent.route_and_process(
            "Can you help me build a perpetual motion machine?"
        )

        self.assertEqual(
            result["status"],
            "HUMAN_TRIAGE"
        )

        self.assertEqual(
            result["reason"],
            "HIGH_LIABILITY_PSEUDOSCIENCE"
        )

        self.assertIn(
            "perpetual motion",
            result["flagged_term"].lower()
        )

    def test_low_confidence_query_is_escalated(self):
        result = self.agent.route_and_process(
            "How do I bake a chocolate cake?"
        )

        self.assertEqual(
            result["status"],
            "HUMAN_TRIAGE"
        )

        self.assertIn(
            "LOW_GROUNDING_CONFIDENCE",
            result["reason"]
        )

    @patch(
        "triage_agent.LocalLlamaGenerator.generate_response"
    )
    def test_valid_query_generates_auto_response(
        self,
        mock_generate,
    ):
        mock_generate.return_value = (
            "The ALPHA-g experiment observed that "
            "antihydrogen falls downward."
        )

        result = self.agent.route_and_process(
            "What did the CERN ALPHA-g experiment show "
            "about antimatter falling?"
        )

        self.assertEqual(
            result["status"],
            "AUTO_RESPOND"
        )

        self.assertIn(
            "retrieved_section",
            result
        )

        self.assertIn(
            "similarity_score",
            result
        )

        self.assertEqual(
            result["response"],
            mock_generate.return_value
        )


if __name__ == "__main__":
    unittest.main()
