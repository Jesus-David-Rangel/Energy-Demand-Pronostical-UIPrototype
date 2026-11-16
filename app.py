import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_and_clean_series, check_stationarity
from src.models import train_and_forecast_arima, train_and_forecast_sarima, forecast_seasonal_naive
from src.evaluation import rolling_origin_evaluation, diagnose_residuals

st.set_page_config(
    page_title="Pronóstico Demanda Eléctrica Colombia - TRL5",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Prototipo Analítico: Pronóstico de Demanda Eléctrica en Colombia")
st.markdown(
    "Sistema de apoyo a la planeación energética basado en modelos de series temporales "
    "(**Seasonal Naive, ARIMA y SARIMA**) sobre registros históricos mensuales (GWh) de **XM / UPME**."
)

# --- Barra lateral de configuración ---
st.sidebar.header("⚙️ Parámetros del Modelo")
uploaded_file = st.sidebar.file_uploader("Cargar serie histórica (CSV)", type=["csv"])
default_path = "data/processed/consumo_energia_colombia.csv"

horizon = st.sidebar.radio("Horizonte de pronóstico (meses)", options=[6, 12], index=1)
model_choice = st.sidebar.selectbox("Modelo principal para proyección", options=["SARIMA", "ARIMA", "Seasonal Naive"])

with st.sidebar.expander("Hiperparámetros (p, d, q) x (P, D, Q, 12)"):
    p = st.number_input("p (AR)", 0, 5, 1)
    d = st.number_input("d (I)", 0, 2, 1)
    q = st.number_input("q (MA)", 0, 5, 1)
    P = st.number_input("P (SAR)", 0, 3, 1)
    D = st.number_input("D (SI)", 0, 2, 1)
    Q = st.number_input("Q (SMA)", 0, 3, 1)

# --- Carga de datos ---
try:
    source = uploaded_file if uploaded_file is not None else default_path
    series, quality_log = load_and_clean_series(source, date_col="fecha", value_col="demanda_gwh")
    adf_res = check_stationarity(series)
except Exception as e:
    st.error(f"No se pudo cargar el archivo de datos. Verifica la ruta o columnas ('fecha', 'demanda_gwh'). Detalle: {e}")
    st.stop()

# --- Pestañas principales del prototipo ---
tab1, tab2, tab3 = st.tabs([
    "📊 1. Calidad de Datos y EDA", 
    "🧪 2. Validación Temporal (Rolling-Origin)", 
    "📈 3. Pronóstico e Incertidumbre"
])

with tab1:
    st.subheader("Bitácora de Calidad y Diagnóstico de Estacionariedad")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observaciones Totales", quality_log["registros_finales"])
    col2.metric("Periodo Cubierto", f"{quality_log['fecha_inicio'][:7]} a {quality_log['fecha_fin'][:7]}")
    col3.metric("Media Mensual (GWh)", f"{quality_log['media_gwh']:,.2f}")
    col4.metric("Prueba ADF (p-value)", adf_res["p_value"], 
                delta="Estacionaria" if adf_res["es_estacionaria_5pct"] else "No Estacionaria (requiere diff)",
                delta_color="normal" if adf_res["es_estacionaria_5pct"] else "inverse")

    fig_eda, ax_eda = plt.subplots(figsize=(12, 4))
    ax_eda.plot(series.index, series.values, color="#1f77b4", linewidth=1.8)
    ax_eda.set_title("Serie Histórica de Consumo Mensual Agregado Nacional (GWh)")
    ax_eda.set_ylabel("Consumo (GWh)")
    ax_eda.grid(True, alpha=0.3)
    st.pyplot(fig_eda)

with tab2:
    st.subheader(f"Evaluación Comparativa fuera de muestra (Horizonte: {horizon} meses)")
    with st.spinner("Ejecutando validación rolling-origin..."):
        eval_df = rolling_origin_evaluation(
            series,
            horizon=horizon,
            n_splits=3 if horizon == 6 else 2,
            arima_order=(p, d, q),
            sarima_order=(p, d, q),
            sarima_seasonal=(P, D, Q, 12)
        )
        summary_df = eval_df.groupby("Modelo")[["MAE", "RMSE", "MAPE (%)"]].mean().reset_index().sort_values("RMSE")

    col_t1, col_t2 = st.columns([1, 1.3])
    with col_t1:
        st.markdown("**Promedio de Métricas por Modelo**")
        st.dataframe(summary_df.style.highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#d4edda"), use_container_width=True)
    with col_t2:
        st.markdown("**Detalle por Ventana de Validación (Splits)**")
        st.dataframe(eval_df, use_container_width=True)

with tab3:
    st.subheader(f"Proyección a {horizon} Meses - Modelo {model_choice}")
    future_dates = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    if model_choice == "SARIMA":
        preds, conf_int, fitted_model = train_and_forecast_sarima(series, horizon, (p, d, q), (P, D, Q, 12))
    elif model_choice == "ARIMA":
        preds, conf_int, fitted_model = train_and_forecast_arima(series, horizon, (p, d, q))
    else:
        preds = forecast_seasonal_naive(series, horizon)
        conf_int, fitted_model = None, None

    # Gráfico de pronóstico
    fig_fc, ax_fc = plt.subplots(figsize=(12, 4.5))
    ax_fc.plot(series.index[-48:], series.values[-48:], label="Histórico (Últimos 48 meses)", color="#1f77b4")
    ax_fc.plot(future_dates, preds, label=f"Pronóstico {model_choice}", color="#d62728", linestyle="--", marker="o")

    df_out = pd.DataFrame({"Fecha": future_dates.strftime("%Y-%m-%d"), "Pronostico_GWh": preds.round(2)})

    if conf_int is not None:
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values
        ax_fc.fill_between(future_dates, lower, upper, color="#d62728", alpha=0.15, label="Intervalo de Confianza 95%")
        df_out["IC_Inferior_95"] = lower.round(2)
        df_out["IC_Superior_95"] = upper.round(2)

    ax_fc.set_title(f"Pronóstico de Demanda Eléctrica Nacional ({horizon} meses)")
    ax_fc.set_ylabel("Demanda (GWh)")
    ax_fc.legend()
    ax_fc.grid(True, alpha=0.3)
    st.pyplot(fig_fc)

    col_res1, col_res2 = st.columns([1.5, 1])
    with col_res1:
        st.dataframe(df_out, use_container_width=True)
        csv_data = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar Pronóstico en CSV",
            data=csv_data,
            file_name=f"pronostico_{model_choice.lower()}_{horizon}m.csv",
            mime="text/csv"
        )
    with col_res2:
        if fitted_model is not None:
            diag = diagnose_residuals(fitted_model)
            st.markdown("**Diagnóstico de Residuos (Ljung-Box)**")
            st.json(diag)
        else:
            st.info("El modelo Seasonal Naive es un benchmark no paramétrico y no genera diagnóstico de residuos.")