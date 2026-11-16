import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def load_and_clean_series(filepath: str, date_col: str, value_col: str) -> tuple[pd.Series, dict]:
    """
    Carga la serie de consumo eléctrico (GWh), valida continuidad mensual,
    trata valores faltantes y genera una bitácora de calidad.
    """
    df = pd.read_csv(filepath)
    raw_count = len(df)
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
    df.set_index(date_col, inplace=True)
    
    # Asegurar frecuencia mensual inicio de mes ('MS') y detectar huecos
    df = df.asfreq('MS')
    missing_count = int(df[value_col].isna().sum())
    
    # Interpolación lineal en caso de valores faltantes aislados
    if missing_count > 0:
        df[value_col] = df[value_col].interpolate(method='linear')
        
    series = df[value_col].astype(float)
    
    quality_log = {
        "registros_originales": raw_count,
        "registros_finales": len(series),
        "fecha_inicio": str(series.index.min().date()),
        "fecha_fin": str(series.index.max().date()),
        "valores_faltantes_imputados": missing_count,
        "media_gwh": round(series.mean(), 2),
        "desviacion_std_gwh": round(series.std(), 2)
    }
    return series, quality_log

def check_stationarity(series: pd.Series) -> dict:
    """Ejecuta la prueba de Dickey-Fuller Aumentada (ADF)."""
    result = adfuller(series.dropna())
    return {
        "adf_statistic": round(result[0], 4),
        "p_value": round(result[1], 4),
        "es_estacionaria_5pct": bool(result[1] < 0.05)
    }