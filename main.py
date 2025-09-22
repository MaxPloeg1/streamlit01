# streamlit_dashboard.py

import json
import numpy as np  # <-- toegevoegd
import pandas as pd
import streamlit as st
import plotly.express as px

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

    # --- FIX: spoor/negatieve neerslagwaarden naar 0 mm ---
    if "RH_mm" in df.columns:
        df["RH_mm"] = pd.to_numeric(df["RH_mm"], errors="coerce").clip(lower=0)

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
avg_temp = df["TG_C"].mean().round(1) if "TG_C" in df.columns else None
total_rain = df["RH_mm"].sum().round(1) if "RH_mm" in df.columns else None
total_sun = df["SQ_h"].sum().round(1) if "SQ_h" in df.columns else None

kpi1, kpi2, kpi3 = st.columns(3)
# --- FIX: toon ook 0-waarden ---
if avg_temp is not None and pd.notna(avg_temp): kpi1.metric("Gemiddelde Temp (°C)", avg_temp)
if total_rain is not None and pd.notna(total_rain): kpi2.metric("Totale Neerslag (mm)", total_rain)
if total_sun is not None and pd.notna(total_sun): kpi3.metric("Totale Zonuren", total_sun)

# === Pagina's ===
if page == "Overzicht":
    st.header("🌍 Jaarlijkse Overzicht")

    yearly = df.groupby("year").agg({
        "TG_C": "mean",
        "RH_mm": "sum",
        "SQ_h": "sum"
    }).reset_index()

    fig = px.bar(
        yearly.melt(id_vars="year", var_name="Maatstaf", value_name="Waarde"),
        x="year", y="Waarde", color="Maatstaf", barmode="group",
        text_auto=".2s",
        title="Vergelijking per jaar (temperatuur, neerslag, zonuren)"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="Waarde", xaxis_title="Jaar")
    st.plotly_chart(fig, use_container_width=True)

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
        # === 1. Boxplot: zonuren per vaste neerslagcategorie ===
        st.subheader("📦 Verdeling zonuren per neerslagcategorie")

        labels = ["0 mm", "0–5 mm", "5–10 mm", "10+ mm"]
        bins = [-0.0001, 0.0001, 5, 10, np.inf]  # exacte 0-bak + open eind

        df["rain_cat"] = pd.cut(
            df["RH_mm"],
            bins=bins, labels=labels,
            right=True, include_lowest=True
        )
        # Forceer categorievolgorde (ook tonen als een bak leeg is)
        df["rain_cat"] = df["rain_cat"].astype(pd.CategoricalDtype(categories=labels, ordered=True))

        fig_box = px.box(
            df, x="rain_cat", y="SQ_h",
            color="rain_cat",
            title="Verdeling zonuren per neerslagcategorie",
            labels={"SQ_h": "Zonuren", "rain_cat": "Neerslagcategorie"},
            points="all",
            category_orders={"rain_cat": labels}
        )
        fig_box.update_traces(marker=dict(opacity=0.5, size=4))
        st.plotly_chart(fig_box, use_container_width=True)

        # === 2. Lijngrafiek: gemiddelde zonuren per dezelfde categorieën ===
        st.subheader("📈 Gemiddelde zonuren bij toenemende neerslag")

        avg_sun = (
            df.groupby("rain_cat", observed=False)["SQ_h"]
              .mean()
              .reindex(labels)  # vaste volgorde en ook lege categorieën behouden
              .reset_index()
        )

        fig_line = px.line(
            avg_sun, x="rain_cat", y="SQ_h",
            markers=True,
            title="Relatie: Neerslagcategorie vs. Gem. zonuren",
            labels={"SQ_h": "Gemiddelde zonuren", "rain_cat": "Neerslagcategorie"},
            category_orders={"rain_cat": labels}
        )
        st.plotly_chart(fig_line, use_container_width=True)

elif page == "Verdeling & Topdagen":
    st.header("📊 Verdeling & Topdagen")

    if "TG_C" in df.columns:
        fig1 = px.box(df, x="season", y="TG_C", points="all",
                      title="Verdeling temperatuur per seizoen")
        st.plotly_chart(fig1, use_container_width=True)

    if "RH_mm" in df.columns:
        top_rain = df.nlargest(10, "RH_mm")[["date", "RH_mm"]]
        fig2 = px.bar(top_rain, x="date", y="RH_mm",
                      title="Top 10 natste dagen", text_auto=".1f", color="RH_mm",
                      color_continuous_scale="Blues")
        st.plotly_chart(fig2, use_container_width=True)

    if "SQ_h" in df.columns:
        top_sun = df.nlargest(10, "SQ_h")[["date", "SQ_h"]]
        fig3 = px.bar(top_sun, x="date", y="SQ_h",
                      title="Top 10 zonnigste dagen", text_auto=".1f", color="SQ_h",
                      color_continuous_scale="Oranges")
        st.plotly_chart(fig3, use_container_width=True)
