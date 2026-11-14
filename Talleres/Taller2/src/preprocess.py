"""Modulo de preprocesamiento de imagenes para reconocimiento de digitos."""

from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps
import torch


def otsu_threshold(gray: np.ndarray) -> int:
    """Calcula el umbral optimo de binarizacion segun el metodo de Otsu.
    
    Maximiza la varianza inter-clase entre fondo y primer plano.
    """
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
        cumulative_sum,
        weight_background,
        out=np.zeros_like(cumulative_sum),
        where=weight_background > 0,
    )
    mean_foreground = np.divide(
        total_sum - cumulative_sum,
        weight_foreground,
        out=np.zeros_like(cumulative_sum),
        where=weight_foreground > 0,
    )
    variance = (
        weight_background
        * weight_foreground
        * (mean_background - mean_foreground) ** 2
    )
    variance[~valid] = -1.0
    return int(np.argmax(variance))


def shift_image(image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """Desplaza una imagen 2D por una cantidad entera de pixeles en X e Y."""
    height, width = image.shape
    shifted = np.zeros_like(image)
    sx0, sx1 = max(0, -shift_x), min(width, width - shift_x)
    sy0, sy1 = max(0, -shift_y), min(height, height - shift_y)
    dx0, dx1 = max(0, shift_x), min(width, width + shift_x)
    dy0, dy1 = max(0, shift_y), min(height, height + shift_y)
    if sx1 > sx0 and sy1 > sy0:
        shifted[dy0:dy1, dx0:dx1] = image[sy0:sy1, sx0:sx1]
    return shifted


def preprocess_image(
    image: Union[Image.Image, np.ndarray, str]
) -> Tuple[torch.Tensor, Image.Image]:
    """Preprocesa una imagen externa o muestra para ser compatible con MNIST y DigitMLP.
    
    Etapas:
        1. Validacion de tipo y dimensiones minimas.
        2. Transposicion EXIF y conversion a escala de grises.
        3. Deteccion de polaridad (fondo claro vs fondo oscuro) e inversion si aplica.
        4. Mejora de contraste adaptativo (autocontrast).
        5. Deteccion de limites del trazo mediante umbral de Otsu adaptado.
        6. Recorte ajustado de la caja delimitadora (bounding box).
        7. Escalado proporcional a maximo 20x20 con interpolacion Lanczos.
        8. Centrado geometrico en lienzo negro de 28x28.
        9. Alineacion por centro de masa ponderado hacia (13.5, 13.5).
        10. Conversion a tensor PyTorch float32 normalizado en [0.0, 1.0].
        
    Returns:
        Tupla con (tensor_1x28x28, imagen_PIL_28x28).
    """
    if image is None:
        raise ValueError("No se recibio ninguna imagen.")

    if isinstance(image, str):
        try:
            image = Image.open(image)
        except Exception as exc:
            raise ValueError(f"No se pudo abrir el archivo de imagen: {exc}") from exc

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("La imagen proporcionada esta vacia.")
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise TypeError("El objeto no es una imagen valida de PIL o matriz NumPy.")

    image = ImageOps.exif_transpose(image).convert("L")
    if image.width < 5 or image.height < 5:
        raise ValueError("La imagen es demasiado pequena para procesarla (minimo 5x5).")

    gray = np.asarray(image, dtype=np.uint8)

    # Deteccion de polaridad en el borde perimetral
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) > 127:
        gray = 255 - gray

    # Verificacion de contraste minimo
    if int(gray.max()) - int(gray.min()) < 15:
        raise ValueError(
            "No se detecto un digito con suficiente contraste. "
            "Use una imagen mas nitida y con el trazo visible."
        )

    # Contraste adaptativo
    gray = np.asarray(
        ImageOps.autocontrast(Image.fromarray(gray), cutoff=1),
        dtype=np.uint8,
    )

    # Umbralizacion adaptativa basada en Otsu
    threshold = max(10, int(otsu_threshold(gray) * 0.5))
    ys, xs = np.where(gray > threshold)
    if len(xs) < 5 or len(ys) < 5:
        raise ValueError(
            "No se detecto un digito con suficiente contraste. "
            "Use una imagen mas nitida y con el trazo visible."
        )

    # Bounding box crop
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = gray[y0:y1, x0:x1]
    crop_h, crop_w = cropped.shape

    # Escalado proporcional a 20x20
    scale = 20.0 / max(crop_h, crop_w)
    new_w = max(1, min(20, int(round(crop_w * scale))))
    new_h = max(1, min(20, int(round(crop_h * scale))))
    resized = Image.fromarray(cropped).resize(
        (new_w, new_h), Image.Resampling.LANCZOS
    )

    # Insercion centrada en lienzo 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = np.asarray(resized)

    # Ajuste fino por centro de masa
    mass = float(canvas.sum())
    if mass <= 0:
        raise ValueError("La imagen no contiene trazos visibles.")
    yy, xx = np.indices(canvas.shape)
    center_x = float((xx * canvas).sum() / mass)
    center_y = float((yy * canvas).sum() / mass)
    canvas = shift_image(
        canvas,
        int(round(13.5 - center_x)),
        int(round(13.5 - center_y)),
    )

    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
    return tensor, Image.fromarray(canvas)


class MNISTCompatibleTransform:
    """Transformador compatible con torchvision Dataset."""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        tensor, _ = preprocess_image(image)
        return tensor
