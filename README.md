# Prototipo Analítico para el Pronóstico de la Demanda Eléctrica en Colombia (TRL5)

**Curso:** Proyecto de Grado (202016907) - Ingeniería de Sistemas (UNAD)  
**Autores:** Jesús David Silva Rangel & Marcial Alberto Díaz Velaides  

**Consulta del proyecto en Streamlit Community Cloud:** 
https://energy-demand-pronostical-uiprototype.streamlit.app/

## Descripción
Prototipo funcional desarrollado en Python para el análisis exploratorio, validación temporal (*rolling-origin*) y pronóstico del consumo mensual agregado nacional de energía eléctrica en Colombia (GWh) a horizontes de 6 y 12 meses mediante modelos **Seasonal Naive**, **ARIMA** y **SARIMA**.

## Ejecución Local
1. Clonar el repositorio e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Despliegue en Streamlit Community Cloud

1. Inicia sesión en [Streamlit Community Cloud](https://share.streamlit.io/) con tu cuenta de GitHub.
2. Selecciona **New app** y el repositorio `Jesus-David-Rangel/Energy-Demand-Pronostical-UIPrototype`.
3. Elige la rama `main` y establece `app.py` como **Main file path**.
4. Pulsa **Deploy**. Streamlit instalará automáticamente `requirements.txt` y usará Python 3.11 mediante `runtime.txt`.

La aplicación utiliza `data/processed/consumo_energia_colombia.csv`, que ya está incluido en el repositorio. Cada actualización de `main` podrá redesplegarse automáticamente desde la configuración de la aplicación.
