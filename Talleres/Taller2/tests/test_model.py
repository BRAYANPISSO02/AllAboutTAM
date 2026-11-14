"""Pruebas unitarias para la arquitectura del modelo DigitMLP."""

import unittest
import torch
from src.model import DigitMLP, create_model


class TestDigitMLP(unittest.TestCase):
    def setUp(self):
        self.model = DigitMLP()

    def test_architecture_structure(self):
        layers = list(self.model.network.children())
        self.assertEqual(len(layers), 6)
        self.assertIsInstance(layers[0], torch.nn.Flatten)
        self.assertIsInstance(layers[1], torch.nn.Linear)
        self.assertEqual(layers[1].in_features, 784)
        self.assertEqual(layers[1].out_features, 64)
        self.assertIsInstance(layers[2], torch.nn.ReLU)
        self.assertIsInstance(layers[3], torch.nn.Linear)
        self.assertEqual(layers[3].in_features, 64)
        self.assertEqual(layers[3].out_features, 1024)
        self.assertIsInstance(layers[4], torch.nn.ReLU)
        self.assertIsInstance(layers[5], torch.nn.Linear)
        self.assertEqual(layers[5].in_features, 1024)
        self.assertEqual(layers[5].out_features, 10)

    def test_forward_4d_tensor(self):
        batch_size = 8
        dummy_input = torch.randn(batch_size, 1, 28, 28)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (batch_size, 10))

    def test_forward_2d_tensor(self):
        batch_size = 4
        dummy_input = torch.randn(batch_size, 784)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (batch_size, 10))

    def test_invalid_dimensions(self):
        # 2D con cantidad erronea de caracteristicas
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 500))

        # 4D con 3 canales en vez de 1
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 3, 28, 28))

        # 4D con resolucion diferente a 28x28
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 1, 32, 32))

        # 3D no soportado
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 28, 28))

    def test_create_model_device(self):
        device = torch.device("cpu")
        model = create_model(device=device)
        first_param = next(model.parameters())
        self.assertEqual(first_param.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
