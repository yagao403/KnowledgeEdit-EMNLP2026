"""Lightweight tests that do not require model weights or a running server."""

import unittest

from core import MODEL_MAP
from core.client import Client
from core.metadata import Metadata
from core.utils import xml_decoder, xml_encoder


class CoreSmokeTests(unittest.TestCase):
    def test_xml_control_character_round_trip(self) -> None:
        value = "text__with\x01control"
        self.assertEqual(xml_decoder(xml_encoder(value)), value)

    def test_vllm_client_configuration(self) -> None:
        client = Client(model="qwen3-32b", base_url="http://localhost:8000")
        self.assertEqual(client.model, "qwen3-32b")
        self.assertEqual(client.base_url, "http://localhost:8000")

    def test_metadata_round_trip(self) -> None:
        metadata = Metadata({"dataset": "fictbio"})
        self.assertEqual(Metadata.from_json(metadata.to_json())["dataset"], "fictbio")

    def test_paper_models_are_registered(self) -> None:
        self.assertIn("qwen3-32b", MODEL_MAP)
        self.assertIn("llama3.1-70b", MODEL_MAP)


if __name__ == "__main__":
    unittest.main()
