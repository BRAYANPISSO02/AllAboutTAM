"""Modulo de evaluacion detallada del modelo sobre el conjunto de prueba."""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.train import evaluate


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = None,
) -> Dict[str, float]:
    """Evalua el modelo sobre el conjunto completo de prueba de 10.000 imagenes."""
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, loader, criterion, device)

    return {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "total_samples": len(loader.dataset),
    }


def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula la matriz de confusion (10x10) y la exactitud por digito individual."""
    if device is None:
        device = next(model.parameters()).device

    confusion = np.zeros((10, 10), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy()
            targets = labels.numpy()

            for target, pred in zip(targets, predictions):
                confusion[target, pred] += 1

    per_class_accuracy = np.diag(confusion) / np.maximum(1, confusion.sum(axis=1))
    return confusion, per_class_accuracy
