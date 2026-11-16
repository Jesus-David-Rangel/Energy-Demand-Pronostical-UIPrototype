import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula MAE, RMSE y MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE (%)": round(float(mape), 2)
    }

def rolling_origin_evaluation(
    series: pd.Series, 
    horizon: int, 
    n_splits: int = 3,
    arima_order: tuple = (1, 1, 1),
    sarima_order: tuple = (1, 1, 1),
    sarima_seasonal: tuple = (1, 1, 1, 12)
) -> pd.DataFrame:
    """
    Validación temporal con origen móvil (rolling-origin) respetando orden cronológico.
    Evalúa Seasonal Naive, ARIMA y SARIMA en múltiples ventanas.
    """
    from src.models import forecast_seasonal_naive, train_and_forecast_arima, train_and_forecast_sarima
    
    results = []
    total_len = len(series)
    
    for i in range(n_splits, 0, -1):
        test_end = total_len - (i - 1) * horizon
        test_start = test_end - horizon
        
        train = series.iloc[:test_start]
        test = series.iloc[test_start:test_end]
        y_true = test.values
        
        # 1. Seasonal Naive
        pred_naive = forecast_seasonal_naive(train, horizon)
        m_naive = calculate_metrics(y_true, pred_naive)
        results.append({"Split": n_splits - i + 1, "Horizonte": horizon, "Modelo": "Seasonal Naive", **m_naive})
        
        # 2. ARIMA
        pred_arima, _, _ = train_and_forecast_arima(train, horizon, order=arima_order)
        m_arima = calculate_metrics(y_true, pred_arima)
        results.append({"Split": n_splits - i + 1, "Horizonte": horizon, "Modelo": "ARIMA", **m_arima})
        
        # 3. SARIMA
        pred_sarima, _, _ = train_and_forecast_sarima(train, horizon, order=sarima_order, seasonal_order=sarima_seasonal)
        m_sarima = calculate_metrics(y_true, pred_sarima)
        results.append({"Split": n_splits - i + 1, "Horizonte": horizon, "Modelo": "SARIMA", **m_sarima})
        
    return pd.DataFrame(results)

def diagnose_residuals(fitted_model) -> dict:
    """Verifica ausencia de autocorrelación residual mediante prueba Ljung-Box."""
    residuals = fitted_model.resid
    lb_test = acorr_ljungbox(residuals, lags=[12], return_df=True)
    p_val = float(lb_test['lb_pvalue'].iloc[0])
    return {
        "ljung_box_pvalue_lag12": round(p_val, 4),
        "ruido_blanco_confirmado": bool(p_val > 0.05)
    }