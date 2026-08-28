"""Ensure the public vLLM server stays limited to the paper's API surface."""

from pathlib import Path
import unittest


class ServerSurfaceTests(unittest.TestCase):
    source = (Path(__file__).parents[1] / "core" / "server.py").read_text(encoding="utf-8")

    def test_only_paper_routes_are_exposed(self) -> None:
        self.assertEqual(self.source.count("@app."), 2)
        self.assertIn('@app.get("/health")', self.source)
        self.assertIn('@app.post("/generate")', self.source)

    def test_non_paper_server_features_are_absent(self) -> None:
        for feature in (
            "/getlogprobs",
            "/update_weight",
            "AsyncLLMEngine",
            "ssl_keyfile",
            "StatelessProcessGroup",
        ):
            self.assertNotIn(feature, self.source)


if __name__ == "__main__":
    unittest.main()
