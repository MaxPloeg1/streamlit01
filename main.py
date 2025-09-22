# streamlit_dashboard.py

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

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

    # Hulpfunctie: schalen
    def make_scaled(src, dest, divisor=10):
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce") / divisor

    # Veelgebruikte kolommen
    make_scaled("TG", "TG_C")
    make_scaled("TN", "TN_C")
    make_scaled("TX", "TX_C")
    make_scaled("RH", "RH_mm")
    make_scaled("EV24", "EV24_mm")
    make_scaled("SQ", "SQ_h")

    # Tijdsfeatures
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week
    df["weekday"] = df["date"].dt.weekday

    def season(m):
        return (
            "winter" if m in [12, 1, 2] else
            "lente"  if m in [3, 4, 5] else
            "zomer"  if m in [6, 7, 8] else
            "herfst"
        )
    df["season"] = df["month"].apply(season)
    return df

# === Sidebar navigatie ===
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Intro", "Temperatuur Trends", "Neerslag & Zon", "Vergelijkingen"]
)

# === Intro pagina ===
if page == "Intro":
    st.title("📊 Amsterdam Weerdata Dashboard")
    st.markdown(
        """
        Welkom bij het interactieve weerdashboard voor Amsterdam.  
        Hier kun je gegevens bekijken zoals temperatuur, neerslag en zonuren.  

        **Stap 1:** Kies een dataset (jaargang).  
        """
    )

    datasets = {
        "2021–2022": "amsterdam_2021_2022.json",
        "2022–2023": "amsterdam_2022_2023.json",
        "2023–2024": "amsterdam_2023_2024.json",
    }

    choice = st.selectbox("📅 Kies een periode:", list(datasets.keys()))
    st.session_state["dataset_choice"] = datasets[choice]

    if choice:
        df_preview = load_data(datasets[choice])
        st.subheader(f"Voorbeeld data ({choice})")
        st.dataframe(df_preview.head())

else:
    # Dataset ophalen uit session_state (gekozen op Intro)
    if "dataset_choice" not in st.session_state:
        st.warning("⚠️ Ga eerst naar de Intro pagina en kies een dataset.")
        st.stop()

    df = load_data(st.session_state["dataset_choice"])

    # === Pagina Temperatuur Trends ===
    if page == "Temperatuur Trends":
        st.title("🌡️ Dagelijkse Temperaturen")
        use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]

        if not use_cols:
            st.error("Geen temperatuurkolommen gevonden")
        else:
            temp = df[["date"] + use_cols].melt("date", var_name="type", value_name="temp_C")

            fig, ax = plt.subplots(figsize=(14, 5))
            sns.lineplot(data=temp, x="date", y="temp_C", hue="type", alpha=0.3, ax=ax)

            for name, sub in temp.groupby("type"):
                sub = sub.sort_values("date").copy()
                sub["roll"] = sub["temp_C"].rolling(7, min_periods=3).mean()
                sns.lineplot(data=sub, x="date", y="roll", label=f"{name} (7d)", ax=ax)

            ax.set_title("Dagelijkse temperatuur (met 7-daagse gemiddelden)")
            ax.set_xlabel("Datum")
            ax.set_ylabel("Temperatuur (°C)")
            st.pyplot(fig)

            # Boxplot per seizoen
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x="season", y="TG_C", ax=ax2)
            ax2.set_title("Verdeling daggemiddelde temperatuur per seizoen")
            st.pyplot(fig2)

    # === Pagina Neerslag & Zon ===
    elif page == "Neerslag & Zon":
        st.title("☔ Neerslag en Zonuren")

        if "RH_mm" in df.columns and "SQ_h" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x="RH_mm", y="SQ_h", hue="season", ax=ax)
            ax.set_xlabel("Neerslag (mm)")
            ax.set_ylabel("Zonuren")
            ax.set_title("Relatie tussen neerslag en zonuren")
            st.pyplot(fig)
        else:
            st.warning("Geen neerslag- of zonurenkolommen beschikbaar")

    # === Pagina Vergelijkingen ===
    elif page == "Vergelijkingen":
        st.title("📊 Jaarlijkse Vergelijkingen")

        yearly = df.groupby("year").agg({
            "TG_C": "mean",
            "RH_mm": "sum",
            "SQ_h": "sum"
        }).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        yearly.plot(x="year", y=["TG_C", "RH_mm", "SQ_h"], kind="bar", ax=ax)
        ax.set_title("Gemiddelde temperatuur en totale neerslag/zon per jaar")
        ax.set_ylabel("Waarde")
        st.pyplot(fig)
