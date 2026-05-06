"""Aplicación Streamlit: Chatbot de Historia del Perú para primaria.

Uso local:
    pip install -r requirements.txt
    export HF_TOKEN='tu_token_opcional'
    streamlit run app_streamlit_historia_peru.py

Uso en Google Colab:
    1) Ejecutar las celdas de instalación y guardado de archivo.
    2) Ejecutar streamlit y exponer puerto con localtunnel/ngrok.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, TypedDict

import streamlit as st
from transformers import pipeline


# -----------------------------
# Configuración y tipos
# -----------------------------

SYSTEM_PROMPT = (
    "Eres un profesor de historia del Perú para estudiantes de primaria. "
    "Responde en español claro, con lenguaje sencillo, ejemplos cortos, "
    "y máximo 120 palabras por respuesta. Si no sabes algo, dilo con honestidad."
)


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class AppConfig:
    """Parámetros de configuración centralizados."""

    model_id: str = "google/flan-t5-base"
    max_new_tokens: int = 160
    temperature: float = 0.4


# -----------------------------
# Carga del modelo de Hugging Face
# -----------------------------

@st.cache_resource(show_spinner=True)
def load_generator(config: AppConfig):
    """Carga el pipeline una sola vez para mejorar rendimiento y costo."""

    hf_token = os.getenv("HF_TOKEN")
    return pipeline(
        task="text2text-generation",
        model=config.model_id,
        token=hf_token,  # opcional para modelos privados o límites más altos
    )


# -----------------------------
# Lógica del chatbot
# -----------------------------

def build_prompt(user_question: str) -> str:
    """Construye un prompt robusto y consistente para el modelo."""

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Instrucciones extra:\n"
        "- Si la pregunta no es de historia del Perú, redirige el tema con amabilidad.\n"
        "- Incluye un dato curioso corto cuando sea posible.\n\n"
        f"Pregunta del estudiante: {user_question}\n"
        "Respuesta del profesor:"
    )


def ask_model(generator, config: AppConfig, user_question: str) -> str:
    """Envía la consulta al pipeline de Hugging Face y devuelve texto limpio."""

    prompt = build_prompt(user_question)
    output = generator(
        prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=True,
    )

    # Para text2text-generation, la llave habitual es generated_text.
    answer = output[0].get("generated_text", "No pude generar una respuesta.")
    return answer.strip()


def init_state() -> None:
    """Inicializa estado de sesión para historial de chat."""

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "¡Hola! Soy tu profe virtual de Historia del Perú. ¿Qué tema te gustaría aprender hoy?",
            }
        ]


def render_chat(messages: List[ChatMessage]) -> None:
    """Renderiza el historial de conversación."""

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main() -> None:
    st.set_page_config(page_title="Chat Historia del Perú", page_icon="📚", layout="centered")
    st.title("📚 Chatbot de Historia del Perú (Primaria)")
    st.caption("Hecho con Python + Hugging Face Pipeline + Streamlit")

    config = AppConfig()
    generator = load_generator(config)
    init_state()
    render_chat(st.session_state.messages)

    user_text = st.chat_input("Escribe tu pregunta (ejemplo: ¿quién fue Túpac Amaru II?)")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Pensando respuesta educativa..."):
            answer = ask_model(generator, config, user_text)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
