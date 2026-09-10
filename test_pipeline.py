"""
Test suite to verify SVI Sigmoid Late Fusion, deterministic hard overrides,
acoustic subscore normalization, and JSON payload contracts.
"""
import math
import sys
import unittest

# Import modules to test
import svi_engine
import call_script
import audio_features


class TestSVIPipeline(unittest.TestCase):

    def test_sigmoid_late_fusion_math(self):
        """Verify sigmoid function behavior at inflection points."""
        # At x = 0.5, sigmoid should be exactly 0.5
        sig_mid = svi_engine.sigmoid_late_fusion(0.5)
        self.assertAlmostEqual(sig_mid, 0.5, places=5)

        # High x should saturate towards 1.0
        sig_high = svi_engine.sigmoid_late_fusion(1.0)
        self.assertGreater(sig_high, 0.98)

        # Low x should saturate towards 0.0
        sig_low = svi_engine.sigmoid_late_fusion(0.0)
        self.assertLess(sig_low, 0.02)

    def test_svi_weights_and_standard_fusion(self):
        """Check standard fusion without hard overrides."""
        # Baseline mild: A(t)=0.2, T(t)=0.2, B(t)=0.1
        raw_svi, delta_crit, category, trigger = svi_engine.compute_svi(
            a_t=0.2, t_t=0.2, b_t=0.1, p_suicide=0.01, p_panic=0.1
        )
        self.assertEqual(delta_crit, 0.0)
        self.assertLess(raw_svi, 30.0)
        self.assertIn(category, ["LOW_RISK", "MODERATE_RISK"])

    def test_deterministic_hard_override_suicide(self):
        """P_suicide > 0.80 must force delta_critical = 0.50 and CRITICAL_RISK (81-100)."""
        raw_svi, delta_crit, category, trigger = svi_engine.compute_svi(
            a_t=0.1, t_t=0.1, b_t=0.0, p_suicide=0.85, p_panic=0.1
        )
        self.assertEqual(delta_crit, 0.50)
        self.assertGreaterEqual(raw_svi, 81.0)
        self.assertLessEqual(raw_svi, 100.0)
        self.assertEqual(category, "CRITICAL_RISK")

    def test_deterministic_hard_override_panic(self):
        """P_panic > 0.80 must force delta_critical = 0.50 and CRITICAL_RISK (81-100)."""
        raw_svi, delta_crit, category, trigger = svi_engine.compute_svi(
            a_t=0.1, t_t=0.1, b_t=0.0, p_suicide=0.01, p_panic=0.88
        )
        self.assertEqual(delta_crit, 0.50)
        self.assertGreaterEqual(raw_svi, 81.0)
        self.assertEqual(category, "CRITICAL_RISK")

    def test_deterministic_hard_override_explicit_marker(self):
        """Explicit markers (rape, weapon, murder) must force delta_critical = 0.50 and CRITICAL_RISK."""
        raw_svi, delta_crit, category, trigger = svi_engine.compute_svi(
            a_t=0.05, t_t=0.05, b_t=0.0, p_suicide=0.01, p_panic=0.1, has_explicit_marker=True
        )
        self.assertEqual(delta_crit, 0.50)
        self.assertGreaterEqual(raw_svi, 81.0)
        self.assertEqual(category, "CRITICAL_RISK")

    def test_ema_smoother(self):
        """EMA smoother update and instant force bypass."""
        smoother = svi_engine.EmaSmoother(initial=10.0, lam=0.3)
        # Update with higher target
        val = smoother.update(50.0)
        # 10 + 0.3 * (50 - 10) = 22.0
        self.assertAlmostEqual(val, 22.0, places=4)

        # Force must snap immediately
        forced = smoother.force(92.0)
        self.assertEqual(forced, 92.0)
        self.assertEqual(smoother.value, 92.0)

    def test_acoustic_subscore_bounds(self):
        """Normalized acoustic subscore A(t) must be bounded in [0.0, 1.0]."""
        # Minimum inputs
        score_min = audio_features.compute_acoustic_subscore(0.0, 0.0, 30.0, 0.0)
        self.assertGreaterEqual(score_min, 0.0)
        self.assertLessEqual(score_min, 0.1)

        # Extreme distress inputs (high jitter, shimmer, f0 variance, 0 HNR)
        score_max = audio_features.compute_acoustic_subscore(5.0, 10.0, 0.0, 50.0)
        self.assertAlmostEqual(score_max, 1.0, places=4)

    def test_scripted_payload_contract_schema(self):
        """Verify each window in SCRIPTED_WINDOWS satisfies the required JSON schema."""
        required_top_keys = {
            "window_id",
            "svi_metrics",
            "acoustic_telemetry",
            "nlp_telemetry",
            "engine_flags",
            "system_actions",
        }
        required_svi_keys = {"score", "category", "primary_trigger"}
        required_acoustic_keys = {"pitch_f0_mean", "jitter", "shimmer_db", "hnr_db"}
        required_nlp_keys = {"transcript", "p_suicide", "p_panic"}
        required_flag_keys = {"bpab", "asd_nse", "slp_akg", "nc_acsm"}
        required_action_keys = {
            "supervisor_barge_in",
            "atss_timer_active",
            "atss_seconds_remaining",
            "witness_protection_prefilled",
            "police_112_dispatch_ready",
        }

        self.assertGreater(len(call_script.SCRIPTED_WINDOWS), 0)

        for w in call_script.SCRIPTED_WINDOWS:
            self.assertTrue(required_top_keys.issubset(w.keys()), f"Missing top keys in window {w}")
            self.assertTrue(required_svi_keys.issubset(w["svi_metrics"].keys()))
            self.assertTrue(required_acoustic_keys.issubset(w["acoustic_telemetry"].keys()))
            self.assertTrue(required_nlp_keys.issubset(w["nlp_telemetry"].keys()))
            self.assertTrue(required_flag_keys.issubset(w["engine_flags"].keys()))
            self.assertTrue(required_action_keys.issubset(w["system_actions"].keys()))
            self.assertIn(w["svi_metrics"]["category"], ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK", "CRITICAL_RISK"])


if __name__ == "__main__":
    unittest.main()
