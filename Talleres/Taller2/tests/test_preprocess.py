"""Pruebas unitarias para el modulo de preprocesamiento de imagenes."""

import unittest
import numpy as np
from PIL import Image, ImageDraw
import torch

from src.preprocess import otsu_threshold, shift_image, preprocess_image


class TestPreprocess(unittest.TestCase):
    def test_otsu_threshold_bimodal(self):
        # Crear distribucion bimodal clara: fondo en 20, objeto en 200
        data = np.zeros((100, 100), dtype=np.uint8) + 20
        data[30:70, 30:70] = 200
        thresh = otsu_threshold(data)
        # El umbral separa fondo (<= t) de objeto (> t); debe estar en [20, 199]
        self.assertTrue(20 <= thresh < 200, f"Umbral Otsu {thresh} fuera de rango")

    def test_shift_image(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5, 5] = 255
        shifted = shift_image(img, shift_x=2, shift_y=-1)
        self.assertEqual(shifted[4, 7], 255)
        self.assertEqual(shifted[5, 5], 0)

    def test_preprocess_dark_background(self):
        # Fondo negro, trazo blanco (como MNIST)
        img = Image.new("L", (100, 100), color=0)
        draw = ImageDraw.Draw(img)
        draw.line([(30, 20), (30, 80)], fill=255, width=8)  # Un trazo vertical tipo "1"
        tensor, pil_img = preprocess_image(img)

        self.assertEqual(tensor.shape, (1, 28, 28))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0)
        self.assertEqual(pil_img.size, (28, 28))

    def test_preprocess_light_background_inversion(self):
        # Fondo blanco (255), trazo negro (0) - tipica foto externa
        img = Image.new("L", (120, 120), color=255)
        draw = ImageDraw.Draw(img)
        draw.line([(40, 20), (40, 100)], fill=0, width=10)
        tensor, pil_img = preprocess_image(img)

        # Tras invertir, el trazo debe ser brillante (>0.5) y los bordes oscuros (~0.0)
        arr = tensor.squeeze().numpy()
        self.assertEqual(tensor.shape, (1, 28, 28))
        # El borde perimetral del canvas debe ser cercano a 0 (fondo)
        border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
        self.assertLess(float(np.mean(border)), 0.1)
        # Debe haber pixeles activos en el centro
        self.assertGreater(float(arr.max()), 0.5)

    def test_preprocess_center_of_mass(self):
        # Imagen con digito descentrado (en la esquina superior izquierda)
        img = Image.new("L", (200, 200), color=255)
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 30, 50], fill=0)
        tensor, _ = preprocess_image(img)

        # Despues de preprocesar, el centro de masa debe estar cerca del centro (13.5, 13.5)
        arr = tensor.squeeze().numpy()
        mass = arr.sum()
        yy, xx = np.indices(arr.shape)
        cx = (xx * arr).sum() / mass
        cy = (yy * arr).sum() / mass
        self.assertAlmostEqual(cx, 13.5, delta=2.5)
        self.assertAlmostEqual(cy, 13.5, delta=2.5)

    def test_preprocess_invalid_inputs(self):
        # None
        with self.assertRaises(ValueError):
            preprocess_image(None)

        # Imagen demasiado pequena
        tiny = Image.new("L", (4, 4), color=0)
        with self.assertRaises(ValueError):
            preprocess_image(tiny)

        # Imagen uniforme sin contraste
        blank = Image.new("L", (50, 50), color=100)
        with self.assertRaises(ValueError):
            preprocess_image(blank)


if __name__ == "__main__":
    unittest.main()
