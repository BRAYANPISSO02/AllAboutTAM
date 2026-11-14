"""Modulo de inferencia y prediccion individual de digitos."""

from typing import Dict, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch import nn

from src.preprocess import preprocess_image


def predict_digit(
    image: Union[Image.Image, np.ndarray, str],
    model: nn.Module,
    device: torch.device = None,
) -> Tuple[int, float, Image.Image, Dict[str, float]]:
    """Ejecuta la inferencia sobre una imagen externa y retorna el resultado detallado.
    
    Args:
        image: Imagen en formato PIL, matriz NumPy o ruta al archivo.
        model: Instancia de DigitMLP cargada en memoria.
        device: Dispositivo de computo (CPU o CUDA).
        
    Returns:
        Tupla con:
            - digit (int): Digito con mayor probabilidad (0-9).
            - confidence (float): Probabilidad normalizada asignada por Softmax.
            - processed_image (PIL.Image): Imagen procesada a 28x28 para feedback visual.
            - probabilities (dict): Mapeo {'0': p0, ..., '9': p9}.
    """
    if device is None:
        device = next(model.parameters()).device

    # Preprocesamiento robusto
    tensor, processed_image = preprocess_image(image)

    # Inferencia con PyTorch
    model.eval()
    with torch.no_grad():
        # Entrada con dimension de lote: [1, 1, 28, 28]
        batch_tensor = tensor.unsqueeze(0).to(device)
        logits = model(batch_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        digit = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[digit].item())

    prob_dict = {str(i): float(probabilities[i].item()) for i in range(10)}

    return digit, confidence, processed_image, prob_dict
