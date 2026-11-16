# Prototipo Analítico para el Pronóstico de la Demanda Eléctrica en Colombia (TRL5)

**Curso:** Proyecto de Grado (202016907) - Ingeniería de Sistemas (UNAD)  
**Autores:** Jesús David Silva Rangel & Marcial Alberto Díaz Velaides  

## Descripción
Prototipo funcional desarrollado en Python para el análisis exploratorio, validación temporal (*rolling-origin*) y pronóstico del consumo mensual agregado nacional de energía eléctrica en Colombia (GWh) a horizontes de 6 y 12 meses mediante modelos **Seasonal Naive**, **ARIMA** y **SARIMA**.

## Ejecución Local
1. Clonar el repositorio e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   python main.py
   streamlit run app.py