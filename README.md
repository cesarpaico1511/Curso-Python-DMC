# Curso-Python-DMC
Ejercicios del curso de Python dictados por DMC. mayo 2026

## Chatbot Historia del Perú (Primaria)

Archivo principal: `app_streamlit_historia_peru.py`

### Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app_streamlit_historia_peru.py
```

### Ejecutar en Google Colab (guía corta)

```python
!pip -q install streamlit transformers torch sentencepiece pyngrok
```

```python
%%writefile app_streamlit_historia_peru.py
# (pegar aquí el contenido completo del archivo del repositorio)
```

```python
from pyngrok import ngrok
import os

# Opcional: token de Hugging Face para evitar límites de rate
os.environ["HF_TOKEN"] = "TU_TOKEN_HF"

public_url = ngrok.connect(8501)
print("URL pública:", public_url)
```

```python
!streamlit run app_streamlit_historia_peru.py --server.port 8501 --server.address 0.0.0.0
```
