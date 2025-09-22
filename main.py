# streamlit_dashboard.py

import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# === Data inlezen ===
@st.cache_data
def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    df = pd.json_normalize(records)

    # Datum parsing
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

    def make_scaled(src, dest, divisor=10):
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce") / divisor

    make_scaled("TG", "TG_C")
    make_scaled("TN", "TN_C")
    make_scaled("TX", "TX_C")
    make_scaled("RH", "RH_mm")
    make_scaled("SQ", "SQ_h")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week

    def season(m):
        return "winter" if m in [12, 1, 2] else "lente" if m in [3, 4, 5] else "zomer" if m in [6, 7, 8] else "herfst"
    df["season"] = df["month"].apply(season)
    return df

# === Sidebar navigatie ===
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Intro", "Overzicht", "Temperatuur Trends", "Neerslag & Zon", "Verdeling & Topdagen"]
)

# === Intro pagina ===
if page == "Intro":
    st.title("📊 Amsterdam Weerdata Dashboard")

    datasets = {
        "2021–2022": "amsterdam_2021_2022.json",
        "2022–2023": "amsterdam_2022_2023.json",
        "2023–2024": "amsterdam_2023_2024.json",
    }

    choice = st.selectbox("📅 Kies een dataset:", list(datasets.keys()))
    st.session_state["dataset_choice"] = datasets[choice]

    if choice:
        df_preview = load_data(datasets[choice])
        st.subheader(f"Voorbeeld data ({choice})")
        st.dataframe(df_preview.head())

else:
    if "dataset_choice" not in st.session_state:
        st.warning("⚠️ Ga eerst naar de Intro pagina en kies een dataset.")
        st.stop()

    df = load_data(st.session_state["dataset_choice"])

    # === KPI-tegels (bovenaan elk dashboard) ===
    avg_temp = df["TG_C"].mean().round(1) if "TG_C" in df else None
    total_rain = df["RH_mm"].sum().round(1) if "RH_mm" in df else None
    total_sun = df["SQ_h"].sum().round(1) if "SQ_h" in df else None

    kpi1, kpi2, kpi3 = st.columns(3)
    if avg_temp: kpi1.metric("Gemiddelde Temp (°C)", avg_temp)
    if total_rain: kpi2.metric("Totale Neerslag (mm)", total_rain)
    if total_sun: kpi3.metric("Totale Zonuren", total_sun)

    # === Overzicht (vergelijkbaar met Channel Comparison) ===
    if page == "Overzicht":
        st.header("🌍 Overzicht per Jaar")
        yearly = df.groupby("year").agg({
            "TG_C": "mean",
            "RH_mm": "sum",
            "SQ_h": "sum"
        }).reset_index()

        fig1 = px.bar(yearly, x="year", y="TG_C", title="Gemiddelde temperatuur per jaar")
        fig2 = px.bar(yearly, x="year", y="RH_mm", title="Totale neerslag per jaar")
        fig3 = px.bar(yearly, x="year", y="SQ_h", title="Totale zonuren per jaar")

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)

    # === Temperatuur Trends (vergelijkbaar met Performance Trends) ===
    elif page == "Temperatuur Trends":
        st.header("🌡️ Temperatuur Trends")
        use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]

        temp = df[["date"] + use_cols].melt("date", var_name="type", value_name="temp_C")
        fig = px.line(temp, x="date", y="temp_C", color="type",
                      title="Dagelijkse temperatuur (min, gem, max)")
        st.plotly_chart(fig, use_container_width=True)

    # === Neerslag & Zon (vergelijkbaar met Shorts Impact) ===
    elif page == "Neerslag & Zon":
        st.header("☔ Neerslag vs. Zon")
        if "RH_mm" in df.columns and "SQ_h" in df.columns:
            fig = px.scatter(df, x="RH_mm", y="SQ_h", color="season",
                             title="Relatie tussen Neerslag en Zonuren",
                             labels={"RH_mm": "Neerslag (mm)", "SQ_h": "Zonuren"})
            st.plotly_chart(fig, use_container_width=True)

    # === Verdeling & Topdagen (vergelijkbaar met Top Videos / Distribution) ===
    elif page == "Verdeling & Topdagen":
        st.header("📊 Verdeling & Topdagen")

        # Boxplot: verdeling temperatuur per seizoen
        if "TG_C" in df:
            fig1 = px.box(df, x="season", y="TG_C", points="all",
                          title="Verdeling daggemiddelde temperatuur per seizoen")
            st.plotly_chart(fig1, use_container_width=True)

        # Top 10 natste dagen
        if "RH_mm" in df:
            top_rain = df.nlargest(10, "RH_mm")[["date", "RH_mm"]]
            fig2 = px.bar(top_rain, x="date", y="RH_mm",
                          title="Top 10 natste dagen", text="RH_mm")
            fig2.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

        # Top 10 zonnigste dagen
        if "SQ_h" in df:
            top_sun = df.nlargest(10, "SQ_h")[["date", "SQ_h"]]
            fig3 = px.bar(top_sun, x="date", y="SQ_h",
                          title="Top 10 zonnigste dagen", text="SQ_h", color_discrete_sequence=["orange"])
            fig3.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)
