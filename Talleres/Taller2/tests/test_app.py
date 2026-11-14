"""Pruebas unitarias para la interfaz Gradio y sus funciones de integracion."""

import unittest
from PIL import Image, ImageDraw
import gradio as gr

from app import classify_image, build_interface, get_model


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Asegurar carga del modelo
        cls.model = get_model()

    def test_classify_none_image(self):
        digit, conf, processed, status = classify_image(None)
        self.assertEqual(digit, "Sin imagen")
        self.assertEqual(conf, "—")
        self.assertIsNone(processed)
        self.assertIn("cargue una imagen", status.lower())

    def test_classify_valid_image(self):
        img = Image.new("L", (100, 100), color=255)
        draw = ImageDraw.Draw(img)
        draw.line([(50, 15), (50, 85)], fill=0, width=8)

        digit, conf, processed, status = classify_image(img)
        self.assertEqual(digit, "1")
        self.assertTrue(conf.endswith("%"))
        self.assertIsNotNone(processed)
        self.assertEqual(processed.size, (28, 28))
        self.assertIn("exitosamente", status.lower())

    def test_classify_blank_image(self):
        blank = Image.new("L", (60, 60), color=128)
        digit, conf, processed, status = classify_image(blank)
        self.assertEqual(digit, "No procesable")
        self.assertEqual(conf, "—")
        self.assertIsNone(processed)
        self.assertIn("Error al procesar", status)

    def test_build_interface_structure(self):
        demo = build_interface()
        self.assertIsInstance(demo, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
