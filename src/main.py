import os
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_and_clean_series, check_stationarity
from src.models import train_and_forecast_sarima
from src.evaluation import rolling_origin_evaluation, diagnose_residuals

def run_pipeline():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    
    # 1. Carga y calidad (ajustar nombres de columnas según archivo real de XM/UPME)
    data_path = "data/raw/consumo_energia_colombia.csv"
    series, quality_log = load_and_clean_series(data_path, date_col="fecha", value_col="demanda_gwh")
    print("=== BITÁCORA DE CALIDAD ===", quality_log)
    
    # 2. Prueba de estacionariedad
    adf_res = check_stationarity(series)
    print("=== PRUEBA ADF ===", adf_res)
    
    # 3. Validación temporal (Horizontes 6 y 12 meses)
    eval_6m = rolling_origin_evaluation(series, horizon=6, n_splits=3)
    eval_12m = rolling_origin_evaluation(series, horizon=12, n_splits=2)
    
    summary = pd.concat([eval_6m, eval_12m]).groupby(["Horizonte", "Modelo"])[["MAE", "RMSE", "MAPE (%)"]].mean().reset_index()
    summary.to_csv("outputs/metrics/comparativa_modelos.csv", index=False)
    print("\n=== COMPARATIVA PROMEDIO DE MODELOS ===\n", summary)
    
    # 4. Entrenamiento final y diagnóstico con modelo ganador (SARIMA)
    forecast_12m, conf_int, fitted_sarima = train_and_forecast_sarima(series, horizon=12)
    diag = diagnose_residuals(fitted_sarima)
    print("\n=== DIAGNÓSTICO DE RESIDUOS ===", diag)
    
    # 5. Exportar pronósticos y gráfico
    future_dates = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq='MS')
    df_forecast = pd.DataFrame({
        "Fecha": future_dates,
        "Pronostico_GWh": forecast_12m.round(2),
        "IC_Inferior_95": conf_int.iloc[:, 0].values.round(2),
        "IC_Superior_95": conf_int.iloc[:, 1].values.round(2)
    })
    df_forecast.to_csv("outputs/metrics/pronostico_12m.csv", index=False)
    
    plt.figure(figsize=(12, 5))
    plt.plot(series.index[-48:], series.values[-48:], label="Histórico (Últimos 4 años)", color="#1f77b4")
    plt.plot(future_dates, forecast_12m, label="Pronóstico SARIMA (12m)", color="#d62728", linestyle="--", marker="o")
    plt.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="#d62728", alpha=0.15, label="IC 95%")
    plt.title("Pronóstico Mensual de Demanda de Energía Eléctrica en Colombia (GWh)")
    plt.xlabel("Fecha")
    plt.ylabel("Demanda (GWh)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figures/pronostico_final_sarima.png", dpi=300)
    print("\nPipeline ejecutado con éxito. Artefactos guardados en /outputs.")

if __name__ == "__main__":
    run_pipeline()