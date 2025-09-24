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

if page == "Overzicht":
    st.header("🌍 Amsterdam: Het Weer in Verandering")
    st.subheader("Warmer – Droger – Zonniger (jaarvergelijking)")

    # === Data voorbereiden ===
    yearly = df.groupby("year").agg({
        "TG_C": "mean",
        "RH_mm": "sum",
        "SQ_h": "sum"
    }).reset_index()

    # Zonuren schalen zodat ze vergelijkbaar zijn met temperatuur
    scale_factor = 200
    yearly["SQ_scaled"] = yearly["SQ_h"] / scale_factor

    # === 1. Hoofdgrafiek: Jaarvergelijking ===
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly["year"], y=yearly["TG_C"],
        name="Gem. Temp (°C)", marker_color="tomato"
    ))
    fig.add_trace(go.Bar(
        x=yearly["year"], y=yearly["SQ_scaled"],
        name=f"Zonuren (x{scale_factor}h)", marker_color="gold"
    ))
    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["RH_mm"],
        name="Neerslag (mm)", mode="lines+markers",
        yaxis="y2", line=dict(color="royalblue")
    ))

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
            f"viel er {diff_rain:.0f} mm minder regen en scheen de zon {diff_sun:.0f} uur langer "
            f"dan in {yearly['year'].iloc[-2]}."
        )

    # === 2. Lange termijn trend temperatuur ===
    avg_yearly_temp = df.groupby("year")["TG_C"].mean().reset_index()
    fig_trend = px.line(
        avg_yearly_temp, x="year", y="TG_C", markers=True,
        title="📈 Lange termijn trend: Gemiddelde jaartemperatuur in Amsterdam",
        labels={"TG_C": "Gemiddelde Temp (°C)", "year": "Jaar"}
    )
    fig_trend.update_traces(line=dict(color="tomato", width=3))
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("👉 Deze grafiek laat zien dat de gemiddelde jaartemperatuur structureel toeneemt, passend bij klimaatopwarming.")

    # === 3. Seizoensgemiddelden ===
    season_temp = df.groupby(["year", "season"])["TG_C"].mean().reset_index()
    fig_season = px.bar(
        season_temp, x="season", y="TG_C", color="year", barmode="group",
        title="🌦️ Gemiddelde temperatuur per seizoen",
        labels={"TG_C": "Gemiddelde Temp (°C)", "season": "Seizoen"}
    )
    st.plotly_chart(fig_season, use_container_width=True)
    st.caption("👉 Vooral de zomers worden warmer – dit merk je direct in hittegolven en langere warme periodes.")

    # === 4. Verdeling zonuren ===
    fig_sun = px.histogram(
        df, x="SQ_h", nbins=30,
        title="☀️ Verdeling van zonuren per dag",
        labels={"SQ_h": "Zonuren per dag", "count": "Aantal dagen"},
        color_discrete_sequence=["gold"]
    )
    st.plotly_chart(fig_sun, use_container_width=True)
    st.caption("👉 Er komen meer dagen met extreem veel zonuren, een teken dat zomers droger en zonniger worden.")

elif page == "Temperatuur Trends":
    st.header("🌡️ Temperatuur Trends")
    use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]

    if use_cols:
        # Mapping van kolomnamen naar labels
        label_map = {"TN_C": "Min temp", "TG_C": "Gem temp", "TX_C": "Max temp"}

        temp = df[["date"] + use_cols].melt("date", var_name="type", value_name="temp_C")
        temp["type"] = temp["type"].replace(label_map)

        fig = px.line(
            temp,
            x="date",
            y="temp_C",
            color="type",
            title="Dagelijkse temperatuur (min, gem, max)",
            labels={"temp_C": "Temperatuur (°C)", "date": "Datum", "type": "Type"},
        )
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

                # Neerslag in categorieën indelen
        rain_bins = pd.cut(
            df["RH_mm"], 
            bins=[0, 1, 5, 10, 20, 50], 
            include_lowest=True, 
            labels=["0 mm", "0–1 mm", "1–5 mm", "5–10 mm", "10–20 mm", "20+ mm"]
        )
        avg_temp_rain = df.groupby(rain_bins)["TG_C"].mean().reset_index()

        # Balkdiagram maken
        fig_temp_rain = px.bar(
            avg_temp_rain, x="RH_mm", y="TG_C",
            title="🌧️ Gemiddelde temperatuur bij toenemende regenval",
            labels={
                "RH_mm": "Neerslagcategorie (mm per dag)", 
                "TG_C": "Gemiddelde temperatuur (°C)"
            },
            text_auto=".1f",  # toon de temperatuurwaarden op de bars
            color="TG_C", 
            color_continuous_scale="RdYlBu_r"
        )

        # Layout verbeteren
        fig_temp_rain.update_layout(
            xaxis_title="Neerslagcategorie (mm per dag)",
            yaxis_title="Gemiddelde temperatuur (°C)",
            showlegend=False
        )

st.plotly_chart(fig_temp_rain, use_container_width=True)


elif page == "Windrichting":
    st.header("💨 Correlatie Windrichting en Seizoenen")   

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
windspeed_seasons.py

Maakt een interactieve boxplot van windsnelheid (FG_ms) per seizoen.
Gebaseerd op KNMI-data met een 'date'-kolom.
"""

import pandas as pd
import plotly.express as px


def add_season_column(df):
    """Voegt een kolom 'season' toe aan de DataFrame gebaseerd op de datum."""
    df['date'] = pd.to_datetime(df['date'])

    def get_season(date):
        month = date.month
        if month in [3, 4, 5]:
            return "Lente"
        elif month in [6, 7, 8]:
            return "Zomer"
        elif month in [9, 10, 11]:
            return "Herfst"
        else:
            return "Winter"

    df['season'] = df['date'].apply(get_season)
    return df


def plot_windspeed_by_season(df):
    """Maakt een interactieve boxplot van windsnelheid per seizoen."""
    season_order = ["Lente", "Zomer", "Herfst", "Winter"]
    season_colors = {
        "Lente": "#2ecc71",
        "Zomer": "#f1c40f",
        "Herfst": "#e67e22",
        "Winter": "#3498db"
    }

    fig = px.box(
        df,
        x="season",
        y="FG_ms",
        color="season",
        category_orders={"season": season_order},
        color_discrete_map=season_colors,
        points="all"
    )

    fig.update_traces(line_width=3)

    fig.update_layout(
        title="Verdeling van windsnelheid per seizoen",
        xaxis_title="Seizoen",
        yaxis_title="Gemiddelde windsnelheid (m/s)",
        title_x=0.5,
        boxmode="group"
    )

    fig.show()


def main():
    # Pas hier het pad naar jouw JSON of CSV-bestand aan
    df = pd.read_json("amsterdam_2023_2024.json")  # of pd.read_csv(...)
    
    # Zet FG om naar m/s als het nog niet gedaan is
    if "FG_ms" not in df.columns and "FG" in df.columns:
        df["FG_ms"] = df["FG"] / 10.0

    df = add_season_column(df)
    plot_windspeed_by_season(df)


if __name__ == "__main__":
    main()

import json
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes


def load_data(json_file):
    """Laad JSON-data en zet om naar DataFrame met FG in m/s."""
    with open(json_file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["FG_ms"] = df["FG"] / 10.0
    assert "FG_ms" in df.columns and "DDVEC" in df.columns, "FG_ms of DDVEC ontbreekt."
    return df


def plot_windrose(df):
    """Maakt een windroos plot van de DataFrame."""
    w = df[["DDVEC", "FG_ms"]].dropna()
    
    ax = WindroseAxes.from_ax()
    ax.bar(
        w["DDVEC"],
        w["FG_ms"],
        normed=True,          # Normaliseer naar fracties
        opening=0.8,
        bins=[0, 2, 4, 6, 8, 10, 12],
        edgecolor="white"
    )
    
    # Y-as in procenten
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"{int(i*100)}%" for i in yticks])
    
    # Titel en legenda
    ax.set_title("Windroos — richting & snelheid (gemiddeld)", pad=30)
    ax.set_legend(title="m/s", loc="center left", bbox_to_anchor=(1.1, 0.5))
    
    plt.tight_layout()
    plt.show()


def main():
    df = load_data("amsterdam_2023_2024.json")
    plot_windrose(df)


if __name__ == "__main__":
    main()

       

