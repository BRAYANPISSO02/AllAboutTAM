"""Pruebas unitarias para el sistema de inferencia y prediccion."""

import unittest
from pathlib import Path
from PIL import Image, ImageDraw
import torch

from src.train import load_trained_model
from src.predict import predict_digit

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "digit_mlp_mnist.pth"


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model = load_trained_model(MODEL_PATH, device=cls.device)

    def test_model_loading_and_weights(self):
        self.assertIsNotNone(self.model)
        self.assertFalse(self.model.training)

    def test_predict_simple_digit_1(self):
        # Crear imagen con digito 1 (trazo vertical centrado)
        img = Image.new("L", (100, 100), color=255)
        draw = ImageDraw.Draw(img)
        draw.line([(50, 15), (50, 85)], fill=0, width=8)

        digit, confidence, processed_img, probs = predict_digit(img, self.model, device=self.device)
        self.assertEqual(digit, 1)
        self.assertGreater(confidence, 0.5)
        self.assertEqual(processed_img.size, (28, 28))
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)

    def test_predict_invalid_image_raises(self):
        blank = Image.new("L", (50, 50), color=200)
        with self.assertRaises(ValueError):
            predict_digit(blank, self.model, device=self.device)


if __name__ == "__main__":
    unittest.main()
