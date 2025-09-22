# streamlit_app.py

# === Imports ===
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid", context="talk")

# === Data inlezen ===
@st.cache_data
def load_data(path="amsterdam_2023_2024.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    df = pd.json_normalize(records)

    # Datum parsen
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

    # Hulpfunctie voor schalen
    def make_scaled(src, dest, divisor=10):
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce") / divisor

    # Temperatuur (°C)
    make_scaled("TG", "TG_C")
    make_scaled("TN", "TN_C")
    make_scaled("TX", "TX_C")
    make_scaled("T10N", "T10N_C")

    # Wind (m/s)
    make_scaled("FG", "FG_ms")
    make_scaled("FXX", "FXX_ms")

    # Windrichting
    if "DDVEC" in df.columns:
        df["DDVEC"] = pd.to_numeric(df["DDVEC"], errors="coerce")

    # Neerslag & verdamping
    make_scaled("RH", "RH_mm")
    make_scaled("EV24", "EV24_mm")

    # Zonduur (uren)
    make_scaled("SQ", "SQ_h")

    # Luchtdruk
    make_scaled("PG", "PG_hPa")
    make_scaled("PX", "PX_hPa")
    make_scaled("PN", "PN_hPa")

    # Tijdsfeatures
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
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

# === Streamlit layout ===
st.title("🌤️ Amsterdam weerdata 2023–2024")

df = load_data()

st.subheader("Voorbeeld van de data")
st.dataframe(df.head())

# === Visualisatie temperatuur ===
use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]
if not use_cols:
    st.error("Geen temperatuurkolommen gevonden (TN/TG/TX).")
else:
    temp = df[["date"] + use_cols].melt("date", var_name="type", value_name="temp_C")

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=temp, x="date", y="temp_C", hue="type", alpha=0.25, ax=ax)

    # 7-daagse rolling mean
    for name, sub in temp.groupby("type"):
        sub = sub.sort_values("date").copy()
        sub["roll"] = sub["temp_C"].rolling(7, min_periods=3).mean()
        sns.lineplot(data=sub, x="date", y="roll", label=f"{name} (7d)", ax=ax)

    ax.set_title("Dagelijkse temperatuur (met 7-daagse gemiddelden)")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Temperatuur (°C)")
    st.pyplot(fig)
