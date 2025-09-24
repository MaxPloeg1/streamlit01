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

# === Sidebar ===
st.sidebar.title("Navigation")

datasets = {
    "2021–2022": "amsterdam_2021_2022.json",
    "2022–2023": "amsterdam_2022_2023.json",
    "2023–2024": "amsterdam_2023_2024.json",
}
dataset_choice = st.sidebar.selectbox("📅 Kies een dataset:", list(datasets.keys()))
df = load_data(datasets[dataset_choice])

page = st.sidebar.radio(
    "Select a page",
    ["Overzicht", "Temperatuur Trends", "Neerslag & Zon", "Verdeling & Topdagen"]
)

# === KPI-tegels ===
avg_temp = df["TG_C"].mean().round(1) if "TG_C" in df else None
total_rain = df["RH_mm"].sum().round(1) if "RH_mm" in df else None
total_sun = df["SQ_h"].sum().round(1) if "SQ_h" in df else None

kpi1, kpi2, kpi3 = st.columns(3)
if avg_temp: kpi1.metric("🌡️ Gemiddelde Temp (°C)", avg_temp)
if total_rain: kpi2.metric("🌧️ Totale Neerslag (mm)", total_rain)
if total_sun: kpi3.metric("☀️ Totale Zonuren", total_sun)

# === Pagina's ===
if page == "Overzicht":
    st.header("🌍 Amsterdam: Het Weer in Verandering")
    st.subheader("Warmer – Droger – Zonniger (jaarvergelijking)")

    yearly = df.groupby("year").agg({
        "TG_C": "mean",
        "RH_mm": "sum",
        "SQ_h": "sum"
    }).reset_index()

    # Zonuren schalen zodat ze vergelijkbaar zijn
    scale_factor = 200
    yearly["SQ_scaled"] = yearly["SQ_h"] / scale_factor

    # Plotly figuur met dubbele as
    fig = go.Figure()

    # Temperatuur
    fig.add_trace(go.Bar(
        x=yearly["year"], y=yearly["TG_C"],
        name="Gem. Temp (°C)", marker_color="tomato"
    ))

    # Zonuren (geschaald)
    fig.add_trace(go.Bar(
        x=yearly["year"], y=yearly["SQ_scaled"],
        name=f"Zonuren (x{scale_factor}h)", marker_color="gold"
    ))

    # Neerslag als lijn
    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["RH_mm"],
        name="Neerslag (mm)", mode="lines+markers", yaxis="y2", line=dict(color="royalblue")
    ))

    # Layout
    fig.update_layout(
        title="Vergelijking per jaar: Temperatuur, Neerslag en Zonuren",
        xaxis_title="Jaar",
        yaxis=dict(title="Temp (°C) & Zonuren (geschaald)", side="left"),
        yaxis2=dict(title="Neerslag (mm)", overlaying="y", side="right"),
        barmode="group",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Praktische conclusie
    if len(yearly) >= 2:
        diff_temp = yearly["TG_C"].iloc[-1] - yearly["TG_C"].iloc[-2]
        diff_rain = yearly["RH_mm"].iloc[-2] - yearly["RH_mm"].iloc[-1]
        diff_sun = yearly["SQ_h"].iloc[-1] - yearly["SQ_h"].iloc[-2]
        st.info(
            f"In {yearly['year'].iloc[-1]} was het gemiddeld {diff_temp:.1f}°C warmer, "
            f"viel er {diff_rain:.0f} mm minder regen en scheen de zon {diff_sun:.0f} uur langer dan in {yearly['year'].iloc[-2]}."
        )

elif page == "Temperatuur Trends":
    st.header("🌡️ Temperatuur Trends")
    use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]

    if use_cols:
        temp = df[["date"] + use_cols].melt("date", var_name="type", value_name="temp_C")
        fig = px.line(temp, x="date", y="temp_C", color="type",
                      title="Dagelijkse temperatuur (min, gem, max)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Neerslag & Zon":
    st.header("☔ Neerslag vs. Zon")

    if "RH_mm" in df.columns and "SQ_h" in df.columns:
        # 1. Boxplot: zonuren per neerslagcategorie
        bins = [0, 1, 5, 10, 50]
        labels = ["0 mm", "0–5 mm", "5–10 mm", "10+ mm"]
        df["rain_cat"] = pd.cut(df["RH_mm"], bins=bins, labels=labels, include_lowest=True)

        fig_box = px.box(
            df, x="rain_cat", y="SQ_h",
            color="rain_cat",
            title="📦 Verdeling zonuren per neerslagcategorie",
            labels={"SQ_h": "Zonuren", "rain_cat": "Neerslagcategorie"},
            points="all"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # 2. Gemiddelde zonuren bij toenemende regen
        rain_bins = pd.cut(df["RH_mm"], bins=20)
        avg_sun = df.groupby(rain_bins)["SQ_h"].mean().reset_index()
        avg_sun["RH_mm"] = avg_sun["RH_mm"].astype(str)

        fig_line = px.line(
            avg_sun, x="RH_mm", y="SQ_h", markers=True,
            title="📈 Gemiddelde zonuren bij toenemende neerslag",
            labels={"SQ_h": "Gemiddelde zonuren", "RH_mm": "Neerslagklasse"}
        )
        st.plotly_chart(fig_line, use_container_width=True)

elif page == "Verdeling & Topdagen":
    st.header("📊 Verdeling & Topdagen")

    if "TG_C" in df:
        fig1 = px.box(df, x="season", y="TG_C", points="all",
                      title="Verdeling temperatuur per seizoen")
        st.plotly_chart(fig1, use_container_width=True)

    if "RH_mm" in df:
        top_rain = df.nlargest(10, "RH_mm")[["date", "RH_mm"]]
        fig2 = px.bar(top_rain, x="date", y="RH_mm",
                      title="Top 10 natste dagen", text_auto=".1f", color="RH_mm",
                      color_continuous_scale="Blues")
        st.plotly_chart(fig2, use_container_width=True)

    if "SQ_h" in df:
        top_sun = df.nlargest(10, "SQ_h")[["date", "SQ_h"]]
        fig3 = px.bar(top_sun, x="date", y="SQ_h",
                      title="Top 10 zonnigste dagen", text_auto=".1f", color="SQ_h",
                      color_continuous_scale="Oranges")
        st.plotly_chart(fig3, use_container_width=True)
