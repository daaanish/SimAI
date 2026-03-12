import unittest
from pathlib import Path


BASE_EXECUTION_TIME_PREDICTOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "vidur"
    / "execution_time_predictor"
    / "base_execution_time_predictor.py"
)


class BaseExecutionTimePredictorFallbackTest(unittest.TestCase):
    """Regression tests for SimAI fallback ordering in TP communication logic."""

    def _verify_fallback_before_assertion(self, backend_section: str):
        fallback_check = "if tensor_parallel_communication_time == -1:"
        non_negative_assertion = "assert tensor_parallel_communication_time >= 0"

        self.assertIn(fallback_check, backend_section)
        self.assertIn(non_negative_assertion, backend_section)
        self.assertLess(
            backend_section.index(fallback_check),
            backend_section.index(non_negative_assertion),
        )

    def test_simai_simulation_fallback_check_precedes_assert(self):
        source = BASE_EXECUTION_TIME_PREDICTOR_PATH.read_text(encoding="utf-8")
        backend_section = source.split('if self._config.backend == "simai_simulation":', 1)[1]
        backend_section = backend_section.split(
            'elif self._config.backend == "simai_analytical":', 1
        )[0]

        self._verify_fallback_before_assertion(backend_section)

    def test_simai_analytical_fallback_check_precedes_assert(self):
        source = BASE_EXECUTION_TIME_PREDICTOR_PATH.read_text(encoding="utf-8")
        backend_section = source.split(
            'elif self._config.backend == "simai_analytical":', 1
        )[1]
        backend_section = backend_section.split(
            'elif self._config.backend == "aicb":', 1
        )[0]

        self._verify_fallback_before_assertion(backend_section)


if __name__ == "__main__":
    unittest.main()
