"""Aplicacion web interactiva con Gradio para reconocimiento de digitos."""

from pathlib import Path
from typing import Optional, Tuple
import gradio as gr
from PIL import Image
import torch

from src.model import DigitMLP
from src.predict import predict_digit
from src.train import load_trained_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "digit_mlp_mnist.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CACHED_MODEL: Optional[DigitMLP] = None


def get_model() -> DigitMLP:
    """Carga el modelo en memoria una sola vez (patron singleton)."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontro el archivo de pesos en '{MODEL_PATH}'. "
                f"Por favor ejecute primero el entrenamiento: python run_train.py"
            )
        _CACHED_MODEL = load_trained_model(MODEL_PATH, device=DEVICE)
    return _CACHED_MODEL


def classify_image(
    image: Optional[Image.Image],
) -> Tuple[str, str, Optional[Image.Image], str]:
    """Callback de inferencia para la interfaz Gradio."""
    if image is None:
        return "Sin imagen", "—", None, "Por favor cargue una imagen para comenzar."

    try:
        model = get_model()
        digit, confidence, processed_image, _ = predict_digit(
            image=image,
            model=model,
            device=DEVICE,
        )
        return (
            str(digit),
            f"{confidence:.2%}",
            processed_image,
            "Prediccion completada exitosamente.",
        )
    except FileNotFoundError as fnf_err:
        return "Error", "—", None, str(fnf_err)
    except Exception as exc:
        return (
            "No procesable",
            "—",
            None,
            f"Error al procesar la imagen: {exc}",
        )


def build_interface() -> gr.Blocks:
    """Construye y configura la interfaz de usuario con Gradio."""
    with gr.Blocks(title="Reconocimiento de Digitos Manuscritos - PyTorch") as demo:
        gr.Markdown(
            """
            # Sistema de Reconocimiento de Digitos Manuscritos
            ### Modelo PyTorch (DigitMLP: 784 -> 64 -> 1024 -> 10) entrenado con MNIST
            
            Cargue una imagen que contenga **un solo digito escrito a mano (0 al 9)**.
            Puede arrastrar y soltar el archivo en la zona de carga o seleccionarlo desde su equipo.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Imagen de entrada (Arrastre o seleccione archivo)",
                    type="pil",
                    sources=["upload"],
                    height=320,
                )
                with gr.Row():
                    recognize_button = gr.Button(
                        "Reconocer digito",
                        variant="primary",
                        size="lg",
                    )
                    clear_button = gr.ClearButton(
                        components=[image_input],
                        value="Limpiar imagen",
                    )

            with gr.Column(scale=1):
                with gr.Row():
                    digit_output = gr.Textbox(
                        label="Digito reconocido",
                        interactive=False,
                        scale=1,
                    )
                    confidence_output = gr.Textbox(
                        label="Nivel de confianza",
                        interactive=False,
                        scale=1,
                    )
                processed_output = gr.Image(
                    label="Imagen preprocesada (28 x 28 px)",
                    type="pil",
                    height=200,
                )
                status_output = gr.Textbox(
                    label="Estado del procesamiento",
                    interactive=False,
                )

        recognize_button.click(
            fn=classify_image,
            inputs=image_input,
            outputs=[
                digit_output,
                confidence_output,
                processed_output,
                status_output,
            ],
        )

        gr.Markdown(
            """
            ---
            **Consideraciones tecnicas:**
            - **Adaptacion de fondo:** El sistema detecta automaticamente si el fondo es claro u oscuro e invierte la imagen para respetar la polaridad de MNIST.
            - **Centrado e invariancia:** El trazo se escala preservando la proporcion y su centro de masa se alinea matematicamente a las coordenadas $(13.5, 13.5)$.
            - **Alcance:** Disenado exclusivamente para digitos individuales. No incluye segmentacion para cifras compuestas.
            - **Confianza:** El porcentaje refleja la salida normalizada por *Softmax* de la red neuronal y es una medida estadistica, no una garantia infalible.
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=theme)
