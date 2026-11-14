"""Modulo de entrenamiento y persistencia del modelo DigitMLP."""

from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model import DigitMLP, create_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Ejecuta una epoca de entrenamiento."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += labels.size(0)

    avg_loss = total_loss / total_items
    accuracy = total_correct / total_items
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evalua el modelo sobre un DataLoader (validacion o prueba)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += labels.size(0)

    avg_loss = total_loss / total_items
    accuracy = total_correct / total_items
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    save_path: Path = None,
    device: torch.device = None,
) -> List[Dict[str, float]]:
    """Ciclo completo de entrenamiento y almacenamiento de pesos."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: List[Dict[str, float]] = []

    print(f"Iniciando entrenamiento en dispositivo: {device}")
    print(f"Parametros: Epocas={epochs}, LR={learning_rate}, BatchSize={train_loader.batch_size}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        history.append(epoch_stats)

        print(
            f"Epoca {epoch:02d}/{epochs} | "
            f"Perdida Train: {train_loss:.4f} | Acc Train: {train_acc:.2%} | "
            f"Perdida Test: {test_loss:.4f} | Acc Test: {test_acc:.2%}"
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": "784-64-1024-10",
                "input_shape": [1, 28, 28],
                "class_names": [str(i) for i in range(10)],
                "epoch": epochs,
                "pixel_range": "[0, 1]",
                "history": history,
            },
            save_path,
        )
        print(f"Modelo guardado exitosamente en: {save_path}")

    return history


def load_trained_model(
    model_path: Path,
    device: torch.device = None,
) -> DigitMLP:
    """Carga un modelo con los pesos guardados desde disco."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de pesos: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = DigitMLP().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
