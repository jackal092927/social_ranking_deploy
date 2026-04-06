import unittest

from lab6.exp1 import run_custom_comparison


class ExperimentSmokeTest(unittest.TestCase):
    def test_small_agg_run(self):
        results, user_scores, W, scoring_rule = run_custom_comparison(
            n_users=12,
            n_influencers=3,
            n_products=6,
            seed=7,
            alpha=0.8,
            beta=0.5,
            metric="agg",
        )

        self.assertEqual(user_scores.shape, (12, 6))
        self.assertEqual(W.shape, (15, 12))
        self.assertEqual(len(scoring_rule), 6)
        self.assertEqual(set(results), {"Random", "Independent", "Sequential", "CAGGRIM"})
        self.assertIn("AggRI", results["CAGGRIM"])

    def test_small_card_run(self):
        results, _, _, _ = run_custom_comparison(
            n_users=12,
            n_influencers=3,
            n_products=6,
            seed=7,
            alpha=0.8,
            beta=0.5,
            metric="card",
        )

        self.assertGreaterEqual(results["CAGGRIM"]["Ratio"], 0)
        self.assertIn("CardRI", results["Sequential"])


if __name__ == "__main__":
    unittest.main()
