import os
import pandas as pd

def prepare_xm_excel():
    os.makedirs("data/processed", exist_ok=True)
    
    # Leer saltando las 2 primeras filas de metadatos de filtros de Power BI
    df = pd.read_excel("data/raw/Demanda de Energía del SIN.xlsx", skiprows=2)
    df.columns = ["Year", "Month", "Day", "Demanda_GWh"]
    
    # Construir fecha diaria y ordenar cronológicamente
    df["fecha_diaria"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-" + df["Day"].astype(str)
    )
    df = df.sort_values("fecha_diaria")
    
    # Agregar a frecuencia mensual (inicio de mes 'MS')
    df_mensual = df.set_index("fecha_diaria")["Demanda_GWh"].resample("MS").sum().reset_index()
    df_mensual.columns = ["fecha", "demanda_gwh"]
    
    # Excluir meses parciales/incompletos (nov-2020 tiene 22 días y nov-2026 tiene 13 días)
    df_mensual = df_mensual[
        (df_mensual["fecha"] >= "2020-12-01") & (df_mensual["fecha"] <= "2026-10-01")
    ].copy()
    
    output_path = "data/processed/consumo_energia_colombia.csv"
    df_mensual.to_csv(output_path, index=False)
    print(f"Dataset procesado guardado en {output_path} ({len(df_mensual)} meses completos).")

if __name__ == "__main__":
    prepare_xm_excel()