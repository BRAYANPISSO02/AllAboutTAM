"""Script para generar y validar imagenes de prueba segun la Seccion 8 del pliego."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.predict import predict_digit
from src.train import load_trained_model

BASE_DIR = Path(__file__).resolve().parent
TEST_IMG_DIR = BASE_DIR / "test_images"
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = BASE_DIR / "models" / "digit_mlp_mnist.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_digit_image(
    digit: int,
    bg_color: int = 255,
    fg_color: int = 0,
    size: tuple = (150, 150),
    stroke_width: int = 8,
    offset: tuple = (0, 0),
) -> Image.Image:
    """Genera una imagen sintetica de un digito con caracteristicas especificas."""
    img = Image.new("L", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Coordenadas aproximadas para dibujar digitos representativos de trazo simple
    cx = size[0] // 2 + offset[0]
    cy = size[1] // 2 + offset[1]
    hw = 25
    hh = 45

    if digit == 0:
        draw.ellipse([cx - hw, cy - hh, cx + hw, cy + hh], outline=fg_color, width=stroke_width)
    elif digit == 1:
        draw.line([(cx, cy - hh), (cx, cy + hh)], fill=fg_color, width=stroke_width)
        draw.line([(cx - 15, cy - hh + 15), (cx, cy - hh)], fill=fg_color, width=stroke_width)
    elif digit == 2:
        draw.arc([cx - hw, cy - hh, cx + hw, cy], start=180, end=0, fill=fg_color, width=stroke_width)
        draw.line([(cx + hw, cy - 5), (cx - hw, cy + hh)], fill=fg_color, width=stroke_width)
        draw.line([(cx - hw, cy + hh), (cx + hw, cy + hh)], fill=fg_color, width=stroke_width)
    elif digit == 3:
        draw.arc([cx - hw, cy - hh, cx + hw, cy], start=270, end=90, fill=fg_color, width=stroke_width)
        draw.arc([cx - hw, cy, cx + hw, cy + hh], start=270, end=90, fill=fg_color, width=stroke_width)
    elif digit == 4:
        draw.line([(cx - hw + 10, cy - hh), (cx - hw, cy + 10)], fill=fg_color, width=stroke_width)
        draw.line([(cx - hw, cy + 10), (cx + hw, cy + 10)], fill=fg_color, width=stroke_width)
        draw.line([(cx + 5, cy - hh), (cx + 5, cy + hh)], fill=fg_color, width=stroke_width)
    elif digit == 7:
        draw.line([(cx - hw, cy - hh), (cx + hw, cy - hh)], fill=fg_color, width=stroke_width)
        draw.line([(cx + hw, cy - hh), (cx - 10, cy + hh)], fill=fg_color, width=stroke_width)
    else:
        # Trazo generico
        draw.line([(cx, cy - hh), (cx, cy + hh)], fill=fg_color, width=stroke_width)

    return img


def run_tests():
    print(f"Cargando modelo desde {MODEL_PATH}...")
    model = load_trained_model(MODEL_PATH, device=DEVICE)

    test_cases = [
        {
            "name": "digito_fondo_blanco_trazo_negro_1.png",
            "desc": "Digito 1 (Fondo blanco, trazo negro, 120x120)",
            "img": create_digit_image(1, bg_color=255, fg_color=0, size=(120, 120), stroke_width=8),
            "expected": 1,
            "type": "valid",
        },
        {
            "name": "digito_fondo_negro_trazo_blanco_0.png",
            "desc": "Digito 0 (Fondo negro, trazo blanco, 100x100)",
            "img": create_digit_image(0, bg_color=0, fg_color=255, size=(100, 100), stroke_width=7),
            "expected": 0,
            "type": "valid",
        },
        {
            "name": "digito_gran_dimension_7.png",
            "desc": "Digito 7 (Dimension grande 300x300, trazo grueso)",
            "img": create_digit_image(7, bg_color=255, fg_color=0, size=(300, 300), stroke_width=16),
            "expected": 7,
            "type": "valid",
        },
        {
            "name": "digito_desplazado_offset_3.png",
            "desc": "Digito 3 (Desplazado a la derecha y abajo)",
            "img": create_digit_image(3, bg_color=255, fg_color=0, size=(160, 160), offset=(30, 25), stroke_width=7),
            "expected": 3,
            "type": "valid",
        },
        {
            "name": "digito_trazo_fino_2.png",
            "desc": "Digito 2 (Trazo fino: 3px)",
            "img": create_digit_image(2, bg_color=255, fg_color=0, size=(120, 120), stroke_width=3),
            "expected": 2,
            "type": "valid",
        },
        {
            "name": "imagen_invalida_sin_contraste.png",
            "desc": "Imagen no compatible: fondo uniforme sin trazo",
            "img": Image.new("L", (80, 80), color=128),
            "expected": "Error",
            "type": "invalid",
        },
        {
            "name": "imagen_invalida_demasiado_pequena.png",
            "desc": "Imagen no compatible: dimension 3x3 px",
            "img": Image.new("L", (3, 3), color=0),
            "expected": "Error",
            "type": "invalid",
        },
    ]

    results = []
    print("\n" + "=" * 90)
    print(f"{'Imagen de prueba':<38} | {'Esperado':<8} | {'Prediccion':<10} | {'Confianza':<10} | {'Resultado':<10}")
    print("=" * 90)

    for case in test_cases:
        file_path = TEST_IMG_DIR / case["name"]
        case["img"].save(file_path)

        if case["type"] == "valid":
            try:
                pred, conf, _, _ = predict_digit(case["img"], model, device=DEVICE)
                passed = (pred == case["expected"])
                res_str = "APROBADO" if passed else "FALLIDO"
                print(f"{case['desc']:<38} | {case['expected']:<8} | {pred:<10} | {conf:.2%}     | {res_str:<10}")
                results.append((case["desc"], case["expected"], str(pred), f"{conf:.2%}", res_str))
            except Exception as e:
                print(f"{case['desc']:<38} | {case['expected']:<8} | ERROR      | —          | FALLIDO ({e})")
                results.append((case["desc"], case["expected"], "ERROR", "—", f"FALLIDO ({e})"))
        else:
            # Caso invalido esperado
            try:
                pred, conf, _, _ = predict_digit(case["img"], model, device=DEVICE)
                print(f"{case['desc']:<38} | Error    | {pred}         | {conf:.2%}     | FALLIDO (Debio fallar)")
                results.append((case["desc"], "Error", str(pred), f"{conf:.2%}", "FALLIDO"))
            except ValueError as ve:
                print(f"{case['desc']:<38} | Error    | Rechazada  | —          | APROBADO (Rechazo correcto)")
                results.append((case["desc"], "Error", "Rechazada", "—", "APROBADO (Control de error)"))
            except Exception as e:
                print(f"{case['desc']:<38} | Error    | Excepcion  | —          | APROBADO ({type(e).__name__})")
                results.append((case["desc"], "Error", "Excepcion", "—", f"APROBADO ({type(e).__name__})"))

    print("=" * 90)
    return results


if __name__ == "__main__":
    run_tests()
