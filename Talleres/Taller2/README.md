# Sistema de Reconocimiento de Dígitos Manuscritos mediante PyTorch

Este proyecto implementa y valida un sistema completo de visión por computadora para el reconocimiento de dígitos manuscritos (0 al 9) basado en el conjunto de datos MNIST, utilizando PyTorch y una interfaz gráfica interactiva en Gradio.

---

## 1. Arquitectura del Modelo (`DigitMLP`)

El modelo respeta la topología de perceptrón multicapa del código de referencia:
* **Entrada:** 784 características ($28 \times 28$ píxeles aplanados).
* **Capa Oculta 1:** 64 neuronas con activación `ReLU`.
* **Capa Oculta 2:** 1024 neuronas con activación `ReLU`.
* **Capa de Salida:** 10 neuronas (logits de clasificación para dígitos 0–9).
* **Pérdida:** `CrossEntropyLoss` (combina LogSoftmax y NLLLoss con estabilidad numérica).
* **Optimizador:** `Adam` ($lr=10^{-3}$).

---

## 2. Preprocesamiento Robusto para Imágenes Externas

El MLP requiere que la imagen externa coincida con la distribución matemática de MNIST:
1. **Detección de polaridad:** Analiza la mediana del borde perimetral. Si es $> 127$ (fondo claro), invierte la imagen para obtener fondo negro y trazo blanco.
2. **Contraste adaptativo:** `ImageOps.autocontrast`.
3. **Umbralización adaptativa:** Algoritmo de **Otsu** para separar el trazo del fondo.
4. **Recorte de Bounding Box:** Aísla la caja mínima que contiene el trazo.
5. **Escalado proporcional a 20 × 20:** Conserva la relación de aspecto original usando interpolación Lanczos.
6. **Centrado geométrico en 28 × 28:** Inserta el trazo en un lienzo de ceros.
7. **Alineación por Centro de Masa:** Calcula el baricentro $(\bar{x}, \bar{y})$ ponderado y desplaza el trazo hacia $(13.5, 13.5)$, cumpliendo la norma original de LeCun.
8. **Normalización:** Tensor `float32` escalado a $[0.0, 1.0]$.

---

## 3. Estructura del Proyecto

```
Taller2/
├── .venv/                         # Entorno virtual con Python 3.11 + PyTorch + CUDA
├── app.py                         # Aplicación Web interactiva (Gradio)
├── data/                          # Dataset MNIST descargado
├── models/                        # Pesos entrenados (digit_mlp_mnist.pth)
├── reconocimiento_digitos.ipynb   # Notebook con las 10 celdas funcionales para Google Colab
├── requirements.txt               # Dependencias del proyecto
├── run_train.py                   # Script CLI para entrenar el modelo (10 épocas)
├── run_evaluate.py                # Script CLI para evaluar sobre 10.000 imágenes de prueba
├── test_specific_recognition.py   # Batería de pruebas específicas con imágenes sintéticas
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Carga de MNIST y DataLoaders
│   ├── evaluate.py                # Métricas, pérdida, accuracy y matriz de confusión
│   ├── model.py                   # Clase DigitMLP y validaciones de dimensiones
│   ├── predict.py                 # Inferencia individual con Softmax y nivel de confianza
│   ├── preprocess.py              # Otsu, shift_image, centrado de masa y pipeline
│   └── train.py                   # Ciclo de entrenamiento y persistencia
└── tests/
    ├── test_app.py                # Pruebas de la interfaz y callbacks
    ├── test_inference.py          # Pruebas de inferencia y predicción
    ├── test_model.py              # Pruebas de arquitectura y dimensionalidad
    └── test_preprocess.py         # Pruebas del pipeline de imágenes y Otsu
```

---

## 4. Instrucciones de Ejecución

### Opción A: Ejecución en Google Colab
1. Abrir Google Colab.
2. Subir el archivo `reconocimiento_digitos.ipynb`.
3. Ejecutar las celdas secuencialmente de la 1 a la 10. En la celda 10 se iniciará la interfaz interactiva Gradio.

### Opción B: Ejecución Local Independiente
1. Activar el entorno virtual:
   ```bash
   source .venv/bin/activate
   ```
2. (Opcional) Si desea reentrenar el modelo:
   ```bash
   python run_train.py
   ```
3. Evaluar el modelo sobre el conjunto de test:
   ```bash
   python run_evaluate.py
   ```
4. Ejecutar la batería de pruebas específicas y unitarias:
   ```bash
   python test_specific_recognition.py
   python -m unittest discover tests
   ```
5. Iniciar la interfaz gráfica:
   ```bash
   python app.py
   ```
   Abra en su navegador `http://127.0.0.1:7860`.
