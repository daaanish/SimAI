import unittest
from pathlib import Path


BASE_EXECUTION_TIME_PREDICTOR = Path(
    "/home/runner/work/SimAI/SimAI/vidur-alibabacloud/vidur/execution_time_predictor/base_execution_time_predictor.py"
)


class BaseExecutionTimePredictorFallbackTest(unittest.TestCase):
    def test_simai_simulation_fallback_check_precedes_assert(self):
        source = BASE_EXECUTION_TIME_PREDICTOR.read_text(encoding="utf-8")
        backend_section = source.split('if self._config.backend == "simai_simulation":', 1)[1]
        backend_section = backend_section.split(
            'elif self._config.backend == "simai_analytical":', 1
        )[0]

        self.assertLess(
            backend_section.index("if tensor_parallel_communication_time == -1:"),
            backend_section.index("assert tensor_parallel_communication_time >= 0"),
        )

    def test_simai_analytical_fallback_check_precedes_assert(self):
        source = BASE_EXECUTION_TIME_PREDICTOR.read_text(encoding="utf-8")
        backend_section = source.split(
            'elif self._config.backend == "simai_analytical":', 1
        )[1]
        backend_section = backend_section.split(
            'elif self._config.backend == "aicb":', 1
        )[0]

        self.assertLess(
            backend_section.index("if tensor_parallel_communication_time == -1:"),
            backend_section.index("assert tensor_parallel_communication_time >= 0"),
        )


if __name__ == "__main__":
    unittest.main()
