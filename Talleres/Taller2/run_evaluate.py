"""Script ejecutable para evaluar el modelo guardado sobre el conjunto de prueba."""

from pathlib import Path
import torch
import numpy as np

from src.dataset import get_dataloaders, get_mnist_datasets
from src.evaluate import evaluate_dataset, compute_confusion_matrix
from src.train import load_trained_model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "digit_mlp_mnist.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=== EVALUACION DEL MODELO DIGITMLP SOBRE TEST (10.000 MUESTRAS) ===")
    print(f"Dispositivo: {DEVICE}")
    print(f"Cargando modelo desde: {MODEL_PATH}")

    model = load_trained_model(MODEL_PATH, device=DEVICE)
    _, test_dataset = get_mnist_datasets(DATA_DIR, use_custom_transform=False)
    _, test_loader = get_dataloaders(
        test_dataset, test_dataset, batch_size=64, num_workers=2, device=DEVICE
    )

    metrics = evaluate_dataset(model, test_loader, device=DEVICE)
    print("\nResultados Globales:")
    print(f"- Total de muestras evaluadas: {metrics['total_samples']:,}")
    print(f"- Perdida media (CrossEntropyLoss): {metrics['test_loss']:.4f}")
    print(f"- Exactitud global (Accuracy): {metrics['test_accuracy']:.2%}")

    confusion, per_class_acc = compute_confusion_matrix(model, test_loader, device=DEVICE)

    print("\nExactitud por digito individual:")
    print("-" * 35)
    print(f"{'Digito':<8} | {'Total':<8} | {'Exactitud':<10}")
    print("-" * 35)
    for digit in range(10):
        total_d = confusion[digit].sum()
        acc_d = per_class_acc[digit]
        print(f"Clase {digit:<2} | {total_d:<8} | {acc_d:.2%}")
    print("-" * 35)


if __name__ == "__main__":
    main()
