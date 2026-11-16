import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

def forecast_seasonal_naive(train: pd.Series, horizon: int, season_length: int = 12) -> np.ndarray:
    """Modelo base ingenuo estacional: repite el mismo mes del año anterior."""
    last_season = train.iloc[-season_length:].values
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season, reps)[:horizon]

def train_and_forecast_arima(
    train: pd.Series, 
    horizon: int, 
    order: tuple = (1, 1, 1)
) -> tuple[np.ndarray, pd.DataFrame, object]:
    """Entrena un modelo ARIMA y retorna pronósticos e intervalos de confianza al 95%."""
    model = SARIMAX(train, order=order, enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    forecast_res = fitted.get_forecast(steps=horizon)
    return forecast_res.predicted_mean.values, forecast_res.conf_int(alpha=0.05), fitted

def train_and_forecast_sarima(
    train: pd.Series, 
    horizon: int, 
    order: tuple = (1, 1, 1), 
    seasonal_order: tuple = (1, 1, 1, 12)
) -> tuple[np.ndarray, pd.DataFrame, object]:
    """Entrena un modelo SARIMA(p,d,q)(P,D,Q,12) y retorna pronósticos e IC al 95%."""
    model = SARIMAX(
        train, 
        order=order, 
        seasonal_order=seasonal_order,
        enforce_stationarity=False, 
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    forecast_res = fitted.get_forecast(steps=horizon)
    return forecast_res.predicted_mean.values, forecast_res.conf_int(alpha=0.05), fitted