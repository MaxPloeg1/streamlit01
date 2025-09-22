# streamlit_dashboard.py

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

sns.set_theme(style="whitegrid")

# === Data inlezen ===
@st.cache_data
def load_data(path="amsterdam_2023_2024.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    df = pd.json_normalize(records)

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
    make_scaled("EV24", "EV24_mm")
    make_scaled("SQ", "SQ_h")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week
    df["weekday"] = df["date"].dt.weekday

    def season(m):
        return "winter" if m in [12,1,2] else "lente" if m in [3,4,5] else "zomer" if m in [6,7,8] else "herfst"
    df["season"] = df["month"].apply(season)
    return df

df = load_data()

# === Sidebar navigatie ===
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["🌡️ Temperatuur Trends", "☔ Neerslag & Zon", "📊 Vergelijkingen"])

# === Pagina 1: Temperatuur Trends ===
if page == "🌡️ Temperatuur Trends":
    st.title("🌡️ Dagelijkse Temperaturen in Amsterdam")
    use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]

    if not use_cols:
        st.error("Geen temperatuurkolommen gevonden")
    else:
        temp = df[["date"] + use_cols].melt("date", var_name="type", value_name="temp_C")

        # Plotly interactieve lijnplot
        fig = px.line(temp, x="date", y="temp_C", color="type",
                      title="Dagelijkse temperatuur per type",
                      labels={"temp_C":"Temperatuur (°C)", "date":"Datum"})
        st.plotly_chart(fig, use_container_width=True)

        # Boxplot verdeling per seizoen
        fig2 = px.box(df, x="season", y="TG_C", points="all",
                      title="Verdeling van daggemiddelde temperatuur per seizoen",
                      labels={"TG_C":"Temperatuur (°C)", "season":"Seizoen"})
        st.plotly_chart(fig2, use_container_width=True)

# === Pagina 2: Neerslag & Zon ===
elif page == "☔ Neerslag & Zon":
    st.title("☔ Neerslag en Zonuren")
    if "RH_mm" in df.columns and "SQ_h" in df.columns:
        fig = px.scatter(df, x="RH_mm", y="SQ_h", color="season",
                         title="Relatie tussen Neerslag en Zonuren",
                         labels={"RH_mm":"Neerslag (mm)", "SQ_h":"Zonuren"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Geen neerslag- of zonurenkolommen beschikbaar")

# === Pagina 3: Vergelijkingen ===
elif page == "📊 Vergelijkingen":
    st.title("📊 Jaarlijkse Vergelijkingen")

    yearly = df.groupby("year").agg({
        "TG_C":"mean",
        "RH_mm":"sum",
        "SQ_h":"sum"
    }).reset_index()

    fig = px.bar(yearly, x="year", y=["TG_C","RH_mm","SQ_h"],
                 title="Gemiddelde temperatuur en totale neerslag/zon per jaar",
                 labels={"value":"Waarde", "year":"Jaar", "variable":"Maatstaf"},
                 barmode="group")
    st.plotly_chart(fig, use_container_width=True)
