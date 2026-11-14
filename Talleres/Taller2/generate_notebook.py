"""Generador del archivo Jupyter Notebook 'reconocimiento_digitos.ipynb'."""

import json
from pathlib import Path

def create_notebook():
    cells = []

    def add_markdown(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header general
    add_markdown("""# Reconocimiento de Dígitos Manuscritos mediante PyTorch

**Proyecto:** Desarrollo, Implementación y Validación de un Sistema de Reconocimiento de Dígitos
**Modelo:** Multi-Layer Perceptron (DigitMLP: 784 → 64 → 1024 → 10)
**Conjunto de datos:** MNIST (Modified National Institute of Standards and Technology)
**Entorno:** Diseñado para ejecutarse secuencialmente en Google Colab o Jupyter Notebook local.

---""")

    # CELDA 1
    add_markdown("""## CELDA 1. Preparación del Entorno, Librerías y Configuración

**Propósito:**
Importa todas las dependencias necesarias, establece semillas de reproducibilidad, verifica la disponibilidad de aceleración por hardware (GPU CUDA o CPU), define hiperparámetros globales y crea las rutas de trabajo para datos y modelos.

**Resultado esperado:**
Información del entorno: versión de PyTorch, dispositivo seleccionado (cuda / cpu) y rutas inicializadas.""")

    add_code("""# Instalacion de dependencias si se ejecuta en Google Colab nuevo:
# !pip install -q torch torchvision gradio pillow matplotlib numpy

import os
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import gradio as gr

# 1. Fijar semillas para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 2. Deteccion de dispositivo (GPU CUDA o CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Configuracion de directorios
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "digit_mlp_mnist.pth"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 4. Hiperparametros de entrenamiento
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 10

print("=== CONFIGURACION DEL ENTORNO ===")
print("PyTorch Version :", torch.__version__)
print("Dispositivo     :", DEVICE, f"({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "")
print("Carpeta Datos   :", DATA_DIR)
print("Ruta Modelo     :", MODEL_PATH)""")

    # CELDA 2
    add_markdown("""## CELDA 2. Carga y Verificación del Conjunto de Datos MNIST

**Propósito:**
Descarga (si no está presente) y carga el conjunto de datos MNIST oficial utilizando torchvision. Inspecciona la cantidad de muestras de entrenamiento y prueba, las dimensiones de las imágenes y la distribución de etiquetas.

**Resultado esperado:**
60.000 imágenes de entrenamiento y 10.000 imágenes de prueba con resolución original de 28 × 28 píxeles.""")

    add_code("""# Carga de las imagenes originales de MNIST en formato PIL/crudo
raw_train_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True
)
raw_test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True
)

print("=== CONJUNTO DE DATOS MNIST ===")
print(f"Muestras de entrenamiento: {len(raw_train_dataset):,}")
print(f"Muestras de prueba       : {len(raw_test_dataset):,}")
sample_pil, sample_label = raw_train_dataset[0]
print(f"Dimensiones de muestra   : {sample_pil.size} (Ancho x Alto en pixeles)")
print(f"Modo de imagen           : {sample_pil.mode}")
print(f"Etiqueta de la muestra   : {sample_label}")""")

    # CELDA 3
    add_markdown("""## CELDA 3. Visualización y Exploración de Datos

**Propósito:**
Genera una cuadrícula visual con muestras aleatorias del conjunto de entrenamiento junto con sus etiquetas verdaderas para comprobar la integridad de los datos y familiarizarse con la variedad caligráfica.

**Resultado esperado:**
Figura de Matplotlib de 2x5 mostrando 10 dígitos manuscritos representativos del 0 al 9.""")

    add_code("""fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for idx, ax in enumerate(axes.flat):
    img, lbl = raw_train_dataset[idx]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Etiqueta: {lbl}", fontsize=11, fontweight="bold")
    ax.axis("off")

plt.suptitle("Muestras representativas del conjunto MNIST", fontsize=14, y=0.98)
plt.tight_layout()
plt.show()""")

    # CELDA 4
    add_markdown("""## CELDA 4. Preprocesamiento Robusto y Preparación de DataLoaders

**Propósito:**
Implementa el canal de transformación matemático que unifica imágenes externas con el formato de MNIST:
1. Umbralización adaptativa con método de **Otsu**.
2. Detección automática de polaridad (fondo claro vs. trazo oscuro) e inversión.
3. Recorte ajustado de la caja delimitadora (*bounding box*).
4. Escalado proporcional preservando relación de aspecto (caja de 20 × 20).
5. Inserción centrada en lienzo negro de 28 × 28 píxeles.
6. Alineación del **centro de masa** hacia las coordenadas $(13.5, 13.5)$ tal como definió LeCun.
7. Construcción de DataLoaders eficientes con `transforms.ToTensor()`.

**Resultado esperado:**
Tensores de forma `[1, 28, 28]` en rango $[0.0, 1.0]$ y DataLoaders listos con 1.875 lotes de entrenamiento y 313 de prueba.""")

    add_code("""def otsu_threshold(gray: np.ndarray) -> int:
    \"\"\"Calcula el umbral optimo de Otsu maximizando la varianza inter-clase.\"\"\"
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    levels = np.arange(256, dtype=np.float64)
    cumulative_count = np.cumsum(histogram)
    cumulative_sum = np.cumsum(histogram * levels)
    total_sum = cumulative_sum[-1]

    weight_background = cumulative_count
    weight_foreground = total - weight_background
    valid = (weight_background > 0) & (weight_foreground > 0)

    mean_background = np.divide(
        cumulative_sum, weight_background,
        out=np.zeros_like(cumulative_sum), where=weight_background > 0
    )
    mean_foreground = np.divide(
        total_sum - cumulative_sum, weight_foreground,
        out=np.zeros_like(cumulative_sum), where=weight_foreground > 0
    )
    variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    variance[~valid] = -1.0
    return int(np.argmax(variance))


def shift_image(image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    \"\"\"Desplaza una matriz 2D con relleno en ceros.\"\"\"
    height, width = image.shape
    shifted = np.zeros_like(image)
    sx0, sx1 = max(0, -shift_x), min(width, width - shift_x)
    sy0, sy1 = max(0, -shift_y), min(height, height - shift_y)
    dx0, dx1 = max(0, shift_x), min(width, width + shift_x)
    dy0, dy1 = max(0, shift_y), min(height, height + shift_y)
    if sx1 > sx0 and sy1 > sy0:
        shifted[dy0:dy1, dx0:dx1] = image[sy0:sy1, sx0:sx1]
    return shifted


def preprocess_image(image):
    \"\"\"Pipeline completo de preprocesamiento para imagenes externas.\"\"\"
    if image is None:
        raise ValueError("No se recibio ninguna imagen.")
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("La imagen esta vacia.")
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("El formato de archivo no es una imagen valida.")

    image = ImageOps.exif_transpose(image).convert("L")
    if image.width < 5 or image.height < 5:
        raise ValueError("La imagen es demasiado pequena para procesarla (minimo 5x5 px).")

    gray = np.asarray(image, dtype=np.uint8)

    # 1. Deteccion automatica de fondo claro vs oscuro
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) > 127:
        gray = 255 - gray

    # 2. Comprobacion de contraste minimo
    if int(gray.max()) - int(gray.min()) < 15:
        raise ValueError("No se detecto un digito con suficiente contraste. Use una imagen mas nitida.")

    # 3. Autocontraste adaptativo
    gray = np.asarray(ImageOps.autocontrast(Image.fromarray(gray), cutoff=1), dtype=np.uint8)

    # 4. Umbralizacion y deteccion de caja delimitadora
    threshold = max(10, int(otsu_threshold(gray) * 0.5))
    ys, xs = np.where(gray > threshold)
    if len(xs) < 5 or len(ys) < 5:
        raise ValueError("No se encontro un trazo reconocible del digito.")

    # 5. Recorte de la region activa
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = gray[y0:y1, x0:x1]
    crop_h, crop_w = cropped.shape

    # 6. Escalado proporcional a maximo 20x20
    scale = 20.0 / max(crop_h, crop_w)
    new_w = max(1, min(20, int(round(crop_w * scale))))
    new_h = max(1, min(20, int(round(crop_h * scale))))
    resized = Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 7. Lienzo negro centrado 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = np.asarray(resized)

    # 8. Ajuste fino por Centro de Masa
    mass = float(canvas.sum())
    if mass <= 0:
        raise ValueError("La imagen no contiene trazos visibles.")
    yy, xx = np.indices(canvas.shape)
    cx = float((xx * canvas).sum() / mass)
    cy = float((yy * canvas).sum() / mass)
    canvas = shift_image(canvas, int(round(13.5 - cx)), int(round(13.5 - cy)))

    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
    return tensor, Image.fromarray(canvas)


# DataLoaders optimizados para entrenamiento
train_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, download=False, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root=str(DATA_DIR), train=False, download=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE.type == "cuda"))

print(f"Lotes de entrenamiento : {len(train_loader)} (tamano {BATCH_SIZE})")
print(f"Lotes de prueba        : {len(test_loader)} (tamano {BATCH_SIZE})")""")

    # CELDA 5
    add_markdown("""## CELDA 5. Arquitectura de la Red Neuronal (DigitMLP)

**Propósito:**
Define la arquitectura neuronal solicitada (MLP `784 → 64 → 1024 → 10` con funciones de activación ReLU). Incluye validaciones explícitas de dimensionalidad tanto para tensores 2D `[lote, 784]` como 4D `[lote, 1, 28, 28]` y produce logits brutos para su consumo por `CrossEntropyLoss`.

**Resultado esperado:**
Modelo instanciado en el dispositivo activo y salida con forma `[4, 10]` para un lote de prueba.""")

    add_code("""class DigitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            if x.shape[1] != 784:
                raise ValueError(f"Entrada 2D invalida: esperaba 784 caracteristicas, recibio {x.shape[1]}.")
        elif x.ndim == 4:
            if tuple(x.shape[1:]) != (1, 28, 28):
                raise ValueError(f"Entrada 4D invalida: esperaba [lote, 1, 28, 28], recibio {tuple(x.shape)}.")
        else:
            raise ValueError(f"Dimension no soportada: ndim={x.ndim}. Use 2D [lote, 784] o 4D [lote, 1, 28, 28].")
        return self.network(x)


model = DigitMLP().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=== ARQUITECTURA DEL MODELO ===")
print(model)
print(f"Total de parametros entrenables: {total_params:,}")

# Verificacion con tensor dummy
with torch.no_grad():
    dummy_out = model(torch.zeros(4, 1, 28, 28, device=DEVICE))
print(f"Forma de salida de prueba: {tuple(dummy_out.shape)} (Lote x Clases)")""")

    # CELDA 6
    add_markdown("""## CELDA 6. Entrenamiento y Almacenamiento del Modelo

**Propósito:**
Entrena el modelo durante 10 épocas utilizando `Adam` ($lr=0.001$) y `CrossEntropyLoss`. Al completar los ciclos, serializa los pesos entrenados y los metadatos en `models/digit_mlp_mnist.pth` para permitir inferencias posteriores sin reentrenar.

**Resultado esperado:**
Pérdida y exactitud registradas por época, y confirmación del guardado del modelo.""")

    add_code("""criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
history = []

print(f"Iniciando entrenamiento ({EPOCHS} epocas)...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_correct, total_items = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += labels.size(0)

    train_loss = total_loss / total_items
    train_acc = total_correct / total_items

    # Evaluacion rapida al final de epoca
    model.eval()
    val_loss, val_correct, val_items = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item() * labels.size(0)
            val_correct += (logits.argmax(dim=1) == labels).sum().item()
            val_items += labels.size(0)

    test_loss = val_loss / val_items
    test_acc = val_correct / val_items

    history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
    print(f"Epoca {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")

# Guardado permanente
torch.save({
    "model_state_dict": model.state_dict(),
    "architecture": "784-64-1024-10",
    "input_shape": [1, 28, 28],
    "class_names": [str(i) for i in range(10)],
    "epoch": EPOCHS,
    "pixel_range": "[0, 1]"
}, MODEL_PATH)

print(f"\\nModelo guardado en: {MODEL_PATH}")""")

    # CELDA 7
    add_markdown("""## CELDA 7. Evaluación Rigurosa del Modelo sobre Test (10.000 Imágenes)

**Propósito:**
Carga los pesos serializados en disco (`torch.load`), evalúa el rendimiento sobre la totalidad del conjunto de prueba y calcula la matriz de confusión y la exactitud desglosada por cada dígito (0 al 9).

**Resultado esperado:**
Exactitud global superior al 97% y desglose por clase de las 10.000 muestras reales.""")

    add_code("""# Carga limpia de pesos desde archivo
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
eval_model = DigitMLP().to(DEVICE)
eval_model.load_state_dict(checkpoint["model_state_dict"])
eval_model.eval()

confusion = np.zeros((10, 10), dtype=np.int64)
total_loss, total_correct, total_samples = 0.0, 0, 0
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = eval_model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1

test_accuracy = total_correct / total_samples
avg_loss = total_loss / total_samples
per_class_accuracy = np.diag(confusion) / confusion.sum(axis=1)

print("=== METRICAS SOBRE CONJUNTO DE PRUEBA (10.000 MUESTRAS) ===")
print(f"Perdida media (Loss) : {avg_loss:.4f}")
print(f"Exactitud global     : {test_accuracy:.2%}")
print("-" * 40)
print(f"{'Digito':<8} | {'Muestras':<10} | {'Exactitud':<10}")
print("-" * 40)
for digit in range(10):
    print(f"Clase {digit:<2} | {confusion[digit].sum():<10} | {per_class_accuracy[digit]:.2%}")
print("-" * 40)""")

    # CELDA 8
    add_markdown("""## CELDA 8. Procesamiento y Verificación de Imágenes Externas

**Propósito:**
Demuestra la capacidad de la función `preprocess_image` para recibir una imagen con características arbitrarias (fondo blanco, dimensiones distintas, traslaciones) y transformarla en el formato canónico de 28 × 28 con alineación por centro de masa.

**Resultado esperado:**
Comparación visual lado a lado entre la imagen original y la procesada a 28 × 28 píxeles.""")

    add_code("""# Demostracion con una muestra del conjunto o externa
demo_pil, demo_target = raw_test_dataset[42]
processed_tensor, processed_pil = preprocess_image(demo_pil)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(demo_pil, cmap="gray")
axes[0].set_title(f"Original (Etiqueta: {demo_target})")
axes[0].axis("off")

axes[1].imshow(processed_pil, cmap="gray")
axes[1].set_title("Procesada (28x28, Centrada)")
axes[1].axis("off")

plt.tight_layout()
plt.show()

print("Forma del tensor procesado :", tuple(processed_tensor.shape))
print("Rango numerico del tensor  :", float(processed_tensor.min()), "a", float(processed_tensor.max()))""")

    # CELDA 9
    add_markdown("""## CELDA 9. Sistema de Inferencia y Predicción Individual

**Propósito:**
Implementa la función `predict_digit(image)` encargada de ejecutar la inferencia completa:
1. Aplica `preprocess_image`.
2. Establece `eval_model.eval()` y desactiva gradientes con `torch.no_grad()`.
3. Calcula las probabilidades de las 10 clases usando `torch.softmax(logits, dim=1)`.
4. Obtiene el dígito de mayor probabilidad y su porcentaje de confianza estimado.

**Resultado esperado:**
Dígito reconocido, nivel de confianza en porcentaje y distribución de probabilidades por clase.""")

    add_code("""def predict_digit(image):
    tensor, processed_img = preprocess_image(image)
    eval_model.eval()
    with torch.no_grad():
        batch_input = tensor.unsqueeze(0).to(DEVICE)
        logits = eval_model(batch_input)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_digit = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_digit].item())

    prob_dict = {str(i): float(probabilities[i].item()) for i in range(10)}
    return predicted_digit, confidence, processed_img, prob_dict


# Prueba con la muestra anterior
pred_digit, pred_conf, _, all_probs = predict_digit(demo_pil)
print("=== RESULTADO DE PREDICCION ===")
print(f"Digito predicho   : {pred_digit} (Esperado: {demo_target})")
print(f"Nivel de confianza: {pred_conf:.2%}")
print("Top 3 probabilidades:")
sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
for digit_str, prob_val in sorted_probs:
    print(f"  Clase {digit_str}: {prob_val:.2%}")""")

    # CELDA 10
    add_markdown("""## CELDA 10. Interfaz Gráfica Interactiva con Gradio

**Propósito:**
Despliega la aplicación web interactiva que cumple con todos los requerimientos obligatorios:
- Arrastrar y soltar (*drag & drop*) y selector de archivos del sistema.
- Procesamiento automático y botón de reconocimiento.
- Visualización de la imagen cargada y de la imagen preprocesada a 28 × 28 píxeles.
- Visualización del dígito predicho y su nivel de confianza.
- Manejo integral de errores con mensajes claros ante imágenes inválidas o sin contraste.
- Conexión directa y real con el modelo de PyTorch.

**Resultado esperado:**
Lanzamiento de la interfaz gráfica accesible localmente o mediante enlace temporal en Google Colab.""")

    add_code("""def gradio_classify(image):
    if image is None:
        return "Sin imagen", "—", None, "Por favor cargue una imagen para comenzar."
    try:
        digit, conf, processed_img, _ = predict_digit(image)
        return str(digit), f"{conf:.2%}", processed_img, "Prediccion completada exitosamente."
    except Exception as exc:
        return "No procesable", "—", None, f"Error al procesar la imagen: {exc}"


with gr.Blocks(title="Reconocimiento de Digitos Manuscritos - PyTorch") as demo:
    gr.Markdown(
        \"\"\"
        # Sistema de Reconocimiento de Digitos Manuscritos
        ### Modelo PyTorch (DigitMLP: 784 → 64 → 1024 → 10) entrenado con MNIST
        
        Cargue una imagen que contenga **un solo digito del 0 al 9**.
        Puede **arrastrar y soltar** el archivo o seleccionarlo desde su equipo.
        \"\"\"
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Imagen de entrada (Arrastre o seleccione)", type="pil", sources=["upload"], height=300)
            with gr.Row():
                recognize_btn = gr.Button("Reconocer digito", variant="primary", size="lg")
                clear_btn = gr.ClearButton(components=[image_input], value="Limpiar")
        with gr.Column(scale=1):
            with gr.Row():
                digit_out = gr.Textbox(label="Digito reconocido", interactive=False)
                conf_out = gr.Textbox(label="Confianza estimada", interactive=False)
            proc_out = gr.Image(label="Imagen preprocesada (28 x 28 px)", type="pil", height=200)
            status_out = gr.Textbox(label="Estado del procesamiento", interactive=False)

    recognize_btn.click(
        fn=gradio_classify,
        inputs=image_input,
        outputs=[digit_out, conf_out, proc_out, status_out]
    )
    gr.Markdown(
        \"\"\"
        ---
        *Nota:* La confianza es una estimacion estadistica de la red neuronal y no garantiza exactitud infalible.
        Disenado exclusivamente para digitos individuales; no incluye segmentacion de numeros de varias cifras.
        \"\"\"
    )

# En Google Colab, puede cambiar a share=True si necesita exponer la interfaz externamente
demo.launch(share=False)""")

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_path = Path("reconocimiento_digitos.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2, ensure_ascii=False)
    print(f"Jupyter Notebook generado exitosamente en: {out_path.resolve()}")

if __name__ == "__main__":
    create_notebook()
