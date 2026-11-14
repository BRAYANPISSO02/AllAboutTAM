"""Arquitectura de la red neuronal para reconocimiento de digitos manuscritos."""

import torch
from torch import nn


class DigitMLP(nn.Module):
    """Perceptron multicapa (MLP) para clasificacion de digitos MNIST.
    
    Arquitectura:
        Entrada: 784 caracteristicas (28x28 aplanado).
        Capa oculta 1: 64 neuronas + ReLU.
        Capa oculta 2: 1024 neuronas + ReLU.
        Capa de salida: 10 neuronas (logits sin normalizar para digitos 0-9).
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagacion hacia adelante con validacion estricta de dimensionalidad.
        
        Args:
            x: Tensor con forma [batch_size, 784] o [batch_size, 1, 28, 28].
            
        Returns:
            Tensor de logits con forma [batch_size, 10].
        """
        if x.ndim == 2:
            if x.shape[1] != 784:
                raise ValueError(
                    f"La entrada aplanada debe tener 784 caracteristicas, pero se recibio {x.shape[1]}."
                )
        elif x.ndim == 4:
            if tuple(x.shape[1:]) != (1, 28, 28):
                raise ValueError(
                    f"La entrada 4D debe tener forma [lote, 1, 28, 28], pero se recibio {tuple(x.shape)}."
                )
        else:
            raise ValueError(
                f"Dimensiones no compatibles: se esperaba tensor 2D [lote, 784] o 4D [lote, 1, 28, 28], recibido ndim={x.ndim}."
            )
        return self.network(x)


def create_model(device: torch.device = None) -> DigitMLP:
    """Instancia el modelo DigitMLP y lo traslada al dispositivo especificado."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitMLP().to(device)
    return model
