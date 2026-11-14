"""Modulo para gestion, descarga y carga del conjunto de datos MNIST."""

from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.preprocess import MNISTCompatibleTransform


def get_mnist_datasets(
    data_dir: Path,
    use_custom_transform: bool = False,
) -> Tuple[datasets.MNIST, datasets.MNIST]:
    """Descarga (si es necesario) y carga los conjuntos de entrenamiento y prueba de MNIST.
    
    Args:
        data_dir: Directorio donde se almacenaran los datos de MNIST.
        use_custom_transform: Si es True, aplica preprocess_image a cada muestra de MNIST.
                              Si es False, aplica transforms.ToTensor() que produce la misma
                              representacion [1, 28, 28] en [0.0, 1.0] de forma altamente optimizada.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if use_custom_transform:
        transform = MNISTCompatibleTransform()
    else:
        transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


def get_dataloaders(
    train_dataset: datasets.MNIST,
    test_dataset: datasets.MNIST,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device = None,
) -> Tuple[DataLoader, DataLoader]:
    """Crea los DataLoaders de PyTorch para entrenamiento y prueba."""
    is_cuda = device is not None and device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )
    return train_loader, test_loader
