"""Script ejecutable para entrenar el modelo DigitMLP sobre MNIST y guardar los pesos."""

import sys
import time
from pathlib import Path
import torch

from src.dataset import get_dataloaders, get_mnist_datasets
from src.model import DigitMLP
from src.train import train_model

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "digit_mlp_mnist.pth"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"=== INICIANDO ENTRENAMIENTO DE DIGITMLP ===")
    print(f"Dispositivo activo: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
    print(f"Directorio de datos: {DATA_DIR}")
    print(f"Ruta de destino del modelo: {MODEL_PATH}")

    # Cargar conjuntos de datos
    train_dataset, test_dataset = get_mnist_datasets(DATA_DIR, use_custom_transform=False)
    train_loader, test_loader = get_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        device=DEVICE,
    )

    model = DigitMLP().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros entrenables: {total_params:,}")

    start_time = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        save_path=MODEL_PATH,
        device=DEVICE,
    )
    elapsed = time.time() - start_time
    print(f"Entrenamiento completado en {elapsed:.2f} segundos.")
    print(f"Ultima epoca - Train Acc: {history[-1]['train_accuracy']:.2%} | Test Acc: {history[-1]['test_accuracy']:.2%}")


if __name__ == "__main__":
    main()
