import streamlit as st
import pandas as pd
import numpy as np
import json
import calendar
import locale
import plotly.express as px

# === Dataset selectie ===
dataset_map = {
    "2021–2022": "amsterdam_2021_2022.json",
    "2022–2023": "amsterdam_2022_2023.json",
    "2023–2024": "amsterdam_2023_2024.json"
}

st.sidebar.header("📂 Datasetkeuze")
dataset_label = st.sidebar.selectbox("📅 Kies een dataset:", list(dataset_map.keys()))
PATH = dataset_map[dataset_label]

# === Data inlezen ===
with open(PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
df = pd.json_normalize(records)

# Datum goedzetten
try:
    df["date"] = pd.to_datetime(df["date"])
except Exception:
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

# Schalen naar normale waarden
def make_scaled(src, dest, divisor=10):
    if src in df.columns:
        df[dest] = pd.to_numeric(df[src], errors="coerce") / divisor

make_scaled("TG", "TG_C")   # Gemiddelde temperatuur
make_scaled("RH", "RH_mm")  # Neerslag
make_scaled("SQ", "SQ_h")   # Zonuren

# Extra tijd-features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# === Sidebar navigatie ===
page = st.sidebar.radio(
    "Navigatie",
    ["🏠 Intro", "📊 Analyses", "🌧️ Neerslag vs Zon", "🌡️ Kalender-heatmap"]
)

# === Pagina’s ===
if page == "🏠 Intro":
    st.title("🌍 Weer Dashboard Amsterdam")
    st.markdown(f"**Geselecteerde dataset:** {dataset_label}")
    st.write("Gebruik de navigatie links om de analyses te bekijken.")

elif page == "📊 Analyses":
    st.header("📊 Tijdreeksen")

    # Selecteer welke maatstaven getoond worden
    opties = []
    if "TG_C" in df.columns: opties.append("Gemiddelde temperatuur (°C)")
    if "RH_mm" in df.columns: opties.append("Neerslag (mm)")
    if "SQ_h" in df.columns: opties.append("Zonuren (h)")

    keuze = st.multiselect("Kies grootheden:", opties, default=opties)

    mapping = {
        "Gemiddelde temperatuur (°C)": "TG_C",
        "Neerslag (mm)": "RH_mm",
        "Zonuren (h)": "SQ_h"
    }

    if keuze:
        kolommen = [mapping[k] for k in keuze]
        fig = px.line(
            df, x="date", y=kolommen,
            labels={"value": "Waarde", "date": "Datum", "variable": "Maatstaf"},
            title="Dagelijkse weerdata"
        )

        # Mooie namen in de legenda
        fig.for_each_trace(lambda t: t.update(
            name=t.name.replace("TG_C", "Gemiddelde temperatuur (°C)")
                         .replace("RH_mm", "Neerslag (mm)")
                         .replace("SQ_h", "Zonuren (h)")
        ))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecteer minstens één grootheid om te plotten.")

elif page == "🌧️ Neerslag vs Zon":
    st.header("☔ Neerslag vs ☀️ Zonuren")
    if "RH_mm" in df.columns and "SQ_h" in df.columns:
        fig = px.scatter(
            df, x="RH_mm", y="SQ_h", color="month",
            labels={"RH_mm": "Neerslag (mm)", "SQ_h": "Zonuren"},
            title="Relatie tussen neerslag en zonuren",
            hover_data=["date"]
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Deze dataset bevat geen neerslag- en zonurenkolommen.")

elif page == "🌡️ Kalender-heatmap":
    st.header("🌡️ Kalender-heatmap temperatuur")

    if "TG_C" in df.columns:
        # Pivot: maand x dag
        pivot = df.pivot_table(index="month", columns="day", values="TG_C", aggfunc="mean")

        # Nederlandse maandnamen
        maanden = {
            1: "januari", 2: "februari", 3: "maart", 4: "april",
            5: "mei", 6: "juni", 7: "juli", 8: "augustus",
            9: "september", 10: "oktober", 11: "november", 12: "december"
        }
        pivot.index = [maanden[m] for m in pivot.index]

        # Heatmap
        fig = px.imshow(
            pivot,
            labels=dict(x="Dag van de maand", y="Maand", color="Gemiddelde temperatuur (°C)"),
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="YlOrRd",
            aspect="auto"
        )

        fig.update_layout(
            title="Kalender-heatmap: Gemiddelde temperatuur per dag",
            xaxis_title="Dag van de maand",
            yaxis_title="Maand"
        )

        # Hover: toon exacte waarde
        fig.update_traces(
            hovertemplate="Dag %{x} %{y}<br>Gem. temp: %{z:.1f} °C<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Geen temperatuurdata (TG_C) beschikbaar in deze dataset.")
