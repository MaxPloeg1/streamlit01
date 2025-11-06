# streamlit_dashboard.py

import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster, HeatMap
import os 

# =========================
# === Data inlezen =========
# =========================
@st.cache_data
def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    df = pd.json_normalize(records)

    # Datum parsing (KNMI dagreeks: 'YYYYMMDD' of al ISO)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        raise ValueError("Kolom 'date' ontbreekt in dataset")

    # Schaal kolommen naar bruikbare eenheden
    def make_scaled(src, dest, divisor=10):
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce") / divisor

    make_scaled("TG", "TG_C")   # etmaaltemp x0.1 °C
    make_scaled("TN", "TN_C")   # minimum temp x0.1 °C
    make_scaled("TX", "TX_C")   # maximum temp x0.1 °C
    make_scaled("RH", "RH_mm")  # neerslag som x0.1 mm
    make_scaled("SQ", "SQ_h")   # zonneschijnduur x0.1 uur

    # Windsnelheid (FG in tienden m/s bij KNMI)
    if "FG" in df.columns:
        df["FG_ms"] = pd.to_numeric(df["FG"], errors="coerce") / 10.0

    # Datum features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear

    # Seizoen
    def season(m):
        if m in [12, 1, 2]:
            return "winter"
        elif m in [3, 4, 5]:
            return "lente"
        elif m in [6, 7, 8]:
            return "zomer"
        else:
            return "herfst"

    df["season"] = df["month"].apply(season)

    # Voor kaart (optionele lat/lon per stad)
    return df


# =========================
# === App layout ==========
# =========================
st.set_page_config(page_title="Weer Dashboard NL", layout="wide")
st.sidebar.title("Navigation")

datasets = {
    "Amsterdam 2021–2022": "amsterdam_2021_2022.json",
    "Amsterdam 2022–2023": "amsterdam_2022_2023.json",
    "Amsterdam 2023–2024": "amsterdam_2023_2024.json",
    "Lauwersoog 2021–2022": "lauwersoog_2021_2022.json",
    "Lauwersoog 2022–2023": "lauwersoog_2022_2023.json",
    "Lauwersoog 2023–2024": "lauwersoog_2023_2024.json",
    "Maastricht 2021–2022": "maastricht_2021_2022.json",
    "Maastricht 2022–2023": "maastricht_2022_2023.json",
    "Maastricht 2023–2024": "maastricht_2023_2024.json",
}
dataset_choice = st.sidebar.selectbox("📅 Kies een dataset:", list(datasets.keys()))
df = load_data(datasets[dataset_choice])

# City afleiden (eerste woord van key)
selected_city = dataset_choice.split()[0]
coords = {
    "Amsterdam": (52.3702, 4.8952),
    "Lauwersoog": (53.4053, 6.2063),
    "Maastricht": (50.8514, 5.6900),
}
# Voeg lat/lon toe per dag (optioneel voor kaart)
if selected_city in coords:
    df["latitude"] = coords[selected_city][0]
    df["longitude"] = coords[selected_city][1]

page = st.sidebar.radio(
    "Select a page",
    ["Overzicht", "Temperatuur Trends", "Neerslag & Zon", "Windtrends & Topdagen", "Voorspelling", "Kaart"]
)

# =========================
# === Overzicht ===========
# =========================
if page == "Overzicht":
    st.header(f"{selected_city}: WeerDashboard Nederland — Overzicht")
    st.markdown("""
Welkom bij ons **WeerDashboard**, waar je in één oogopslag het weer in verschillende Nederlandse steden kunt verkennen en begrijpen.  
We analyseren gegevens uit **Amsterdam**, **Lauwersoog** en **Maastricht** over de periodes **2021–2024** om trends en patronen in temperatuur, neerslag, zonuren en wind zichtbaar te maken.  

###Inhoud van het dashboard:
- **Temperatuur Trends:** ontdek hoe temperaturen zich door de jaren heen hebben ontwikkeld.  
- **Neerslag & Zon:** vergelijk regenval met het aantal zonuren per seizoen of jaar.  
- **Windtrends & Topdagen:** bekijk de krachtigste winddagen en de invloed van wind op het weerbeeld.  
- **Voorspelling:** gebruik onze voorspelfunctie om de temperatuur te schatten op basis van zonuren, neerslag en wind.  
- **Kaart:** verken regionale verschillen in weerpatronen op een interactieve kaart.  

Of je nu geïnteresseerd bent in klimaatverandering, planning van buitenactiviteiten of gewoon nieuwsgierig bent naar het Nederlandse weer —  
dit dashboard biedt **inzicht, overzicht en voorspelling in één**.
""")

# =========================
# === Temperatuur Trends ==
# =========================
elif page == "Temperatuur Trends":
    st.header("🌡️ Temperatuur Trends")
    use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]

    if use_cols:
        label_map = {"TN_C": "Min temp", "TG_C": "Gem temp", "TX_C": "Max temp"}
        temp = df[["date"] + use_cols].melt("date", var_name="type", value_name="temp_C")
        temp["type"] = temp["type"].replace(label_map)

        fig = px.line(
            temp, x="date", y="temp_C", color="type",
            title="Dagelijkse temperatuur (min, gem, max)",
            labels={"temp_C": "Temperatuur (°C)", "date": "Datum", "type": "Type"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Boxplot temperatuur per maand
        df["month_name"] = df["date"].dt.month_name()

        fig_box = px.box(
            df, x="month_name", y="TG_C",
            category_orders={"month_name": [
                "January","February","March","April","May","June",
                "July","August","September","October","November","December"
            ]},
            title="📦 Verdeling van gemiddelde temperatuur per maand",
            labels={"month_name": "Maand", "TG_C": "Gemiddelde temperatuur (°C)"},
            color="month_name"
        )
        fig_box.update_traces(line_width=2)
        st.plotly_chart(fig_box, use_container_width=True)

        # Gemiddelde temperatuur per seizoen
        season_temp = df.groupby("season")["TG_C"].mean().reset_index()
        season_colors = {"winter": "#3498db", "lente": "#2ecc71", "zomer": "#f1c40f", "herfst": "#e67e22"}

        fig_season = px.bar(
            season_temp, x="season", y="TG_C", color="season",
            title="🌦️ Gemiddelde temperatuur per seizoen",
            labels={"season": "Seizoen", "TG_C": "Gemiddelde Temp (°C)"},
            color_discrete_map=season_colors
        )
        st.plotly_chart(fig_season, use_container_width=True)
    else:
        st.info("Temperatuurkolommen (TN_C/TG_C/TX_C) niet gevonden in dataset.")


# =========================
# === Neerslag & Zon ======
# =========================
elif page == "Neerslag & Zon":
    st.header("☔ Neerslag vs. Zon")

    if "RH_mm" in df.columns and "SQ_h" in df.columns:
        bins = [0, 1, 5, 10, 50]
        labels = ["0 mm", "0–5 mm", "5–10 mm", "10+ mm"]
        df["rain_cat"] = pd.cut(df["RH_mm"], bins=bins, labels=labels, include_lowest=True)
        ordered_cats = ["0 mm", "0–5 mm", "5–10 mm", "10+ mm"]
        df["rain_cat"] = pd.Categorical(df["rain_cat"], categories=ordered_cats, ordered=True)

        color_map = {
            "0 mm": "#d7263d",
            "0–5 mm": "#0e6eb8",
            "5–10 mm": "#74a9cf",
            "10+ mm": "#eab0b6"
        }

        fig_box = px.box(
            df, x="rain_cat", y="SQ_h", color="rain_cat",
            category_orders={"rain_cat": ordered_cats},
            color_discrete_map=color_map,
            title="📦 Verdeling zonuren per neerslagcategorie",
            labels={"SQ_h": "Zonuren", "rain_cat": "Neerslagcategorie"},
            points="all"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        rain_bins = pd.cut(
            df["RH_mm"], bins=[0, 1, 5, 10, 20, 50],
            include_lowest=True,
            labels=["0–1 mm", "1–5 mm", "5–10 mm", "10–20 mm", "20+ mm"]
        )
        avg_temp_rain = df.groupby(rain_bins)["TG_C"].mean().reset_index()

        fig_temp_rain = px.bar(
            avg_temp_rain, x="RH_mm", y="TG_C",
            title="🌧️ Gemiddelde temperatuur bij toenemende regenval",
            labels={"RH_mm": "Neerslagcategorie (mm per dag)", "TG_C": "Gemiddelde temperatuur (°C)"},
            text_auto=".1f",
            color="TG_C",
            color_continuous_scale="RdYlBu_r"
        )
        fig_temp_rain.update_layout(
            xaxis_title="Neerslagcategorie (mm per dag)",
            yaxis_title="Gemiddelde temperatuur (°C)",
            showlegend=False
        )
        st.plotly_chart(fig_temp_rain, use_container_width=True)
    else:
        st.info("RH_mm en/of SQ_h ontbreken in dataset.")


# =========================
# === Windtrends & Topdagen
# =========================
elif page == "Windtrends & Topdagen":
    st.header("📊 Windtrends & Topdagen")

    # 1) Kalender-heatmap temperatuur
    if "TG_C" in df.columns and "date" in df.columns:
        tmp = df.copy()
        tmp["day"] = tmp["date"].dt.day
        pivot = tmp.pivot_table(index="month", columns="day", values="TG_C", aggfunc="mean")
        # Vul dagen 1..31 zodat imshow geen NaN-only kolommen mist
        all_days = list(range(1, 32))
        pivot = pivot.reindex(columns=all_days)
        month_names = {
            1: "Januari", 2: "Februari", 3: "Maart", 4: "April",
            5: "Mei", 6: "Juni", 7: "Juli", 8: "Augustus",
            9: "September", 10: "Oktober", 11: "November", 12: "December"
        }
        pivot.index = pivot.index.map(month_names)

        fig_heatmap = px.imshow(
            pivot,
            color_continuous_scale="RdBu_r",
            origin="upper",
            aspect="auto",
            labels=dict(color="Temperatuur (°C)", x="Dag van de maand", y="Maand")
        )
        fig_heatmap.update_xaxes(title="Dag van de maand", tickmode="linear")
        fig_heatmap.update_yaxes(title="Maand", tickmode="array",
                                 tickvals=list(pivot.index), ticktext=list(pivot.index))
        fig_heatmap.update_layout(title="📅 Kalender-heatmap: gemiddelde temperatuur per dag", title_x=0.5)

        st.plotly_chart(fig_heatmap, use_container_width=True)

    # 2) Windroos
    if "FG_ms" in df.columns and "DDVEC" in df.columns:
        try:
            import matplotlib.pyplot as plt
            from windrose import WindroseAxes  # noqa: F401

            w = df[["DDVEC", "FG_ms"]].dropna()
            fig_wind = plt.figure(figsize=(6, 6))
            ax = fig_wind.add_subplot(111, projection="windrose")
            ax.bar(
                w["DDVEC"],
                w["FG_ms"],
                normed=True,
                opening=0.8,
                bins=[0, 2, 4, 6, 8, 10, 12],
                edgecolor="white"
            )
            ax.set_title("Windroos — richting & snelheid (gemiddeld)", pad=20)
            ax.set_legend(title="m/s", loc="center left", bbox_to_anchor=(1.1, 0.5))
            st.pyplot(fig_wind)
        except Exception as e:
            st.info("Windroos vereist het pakket 'windrose'. Installeer met: `pip install windrose`.")
    else:
        st.info("Benodigde kolommen voor windroos ontbreken (FG_ms en DDVEC).")

    # 3) Boxplot windsnelheid per seizoen
    if "FG_ms" in df.columns and "date" in df.columns:
        st.subheader("📦 Verdeling van windsnelheid per seizoen")

        def get_season_name(date):
            m = date.month
            if m in [3, 4, 5]:
                return "Lente"
            elif m in [6, 7, 8]:
                return "Zomer"
            elif m in [9, 10, 11]:
                return "Herfst"
            else:
                return "Winter"

        df['season_box'] = df['date'].apply(get_season_name)

        season_order = ["Lente", "Zomer", "Herfst", "Winter"]
        season_colors = {
            "Lente": "#2ecc71",
            "Zomer": "#f1c40f",
            "Herfst": "#e67e22",
            "Winter": "#3498db"
        }

        fig_box = px.box(
            df,
            x="season_box",
            y="FG_ms",
            color="season_box",
            category_orders={"season_box": season_order},
            color_discrete_map=season_colors,
            points="all"
        )
        fig_box.update_traces(line_width=3)
        fig_box.update_layout(
            title="📦 Verdeling windsnelheid per seizoen",
            xaxis_title="Seizoen",
            yaxis_title="Gemiddelde windsnelheid (m/s)",
            title_x=0.5,
            boxmode="group"
        )
        st.plotly_chart(fig_box, use_container_width=True)


# =========================
# === Voorspelling =========
# =========================
elif page == "Voorspelling":
    st.header("🤖 Voorspellend model: Temperatuur voorspellen")

    # Controleer noodzakelijke kolommen
    needed_cols_any = ["TG_C"]
    if not all(c in df.columns for c in needed_cols_any):
        st.warning("Kolom 'TG_C' ontbreekt; kan geen temperatuurmodel trainen.")
    else:
        # Feature engineering
        work = df.copy()

        # Basishulpkolommen voor seizoenspatroon (cyclische encodings)
        # sin/cos over dag-van-het-jaar en maand
        work["sin_doy"] = np.sin(2 * np.pi * work["dayofyear"] / 365.25)
        work["cos_doy"] = np.cos(2 * np.pi * work["dayofyear"] / 365.25)
        work["sin_mon"] = np.sin(2 * np.pi * work["month"] / 12.0)
        work["cos_mon"] = np.cos(2 * np.pi * work["month"] / 12.0)

        # Beschikbare meteorologische features
        candidate_features = []
        for col in ["RH_mm", "SQ_h", "FG_ms"]:
            if col in work.columns:
                candidate_features.append(col)

        # Friendly names for features so the UI shows clear labels instead of raw column names
        friendly_feature_names = {
            "RH_mm": "Neerslag (mm)",
            "SQ_h": "Zonuren (uur)",
            "FG_ms": "Windsnelheid (m/s)"
        }

        # Multiselect voor features (toont vriendelijke namen, maar gebruikt de kolomnamen intern)
        default_feats = ["RH_mm", "SQ_h", "FG_ms"]
        default_feats = [f for f in default_feats if f in candidate_features]
        feature_choice = st.multiselect(
            "Kies extra invoer-variabelen (optioneel):",
            options=candidate_features,
            format_func=lambda x: friendly_feature_names.get(x, x),
            default=default_feats,
            help="Selecteer hier welke externe variabelen (regen/zon/wind) je wilt gebruiken naast de seizoensfeatures."
        )

        # Definitieve feature set
        features = ["sin_doy", "cos_doy", "sin_mon", "cos_mon"] + feature_choice
        work = work.dropna(subset=["TG_C"] + [c for c in feature_choice if c in work.columns])

        X = work[features]
        y = work["TG_C"]

        # Modelkeuze en parameters
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        model_choice = st.selectbox("Kies modeltype:", ["Lineaire Regressie", "Random Forest"])

        if model_choice == "Lineaire Regressie":
            from sklearn.linear_model import LinearRegression
            # LinearRegression kan baat hebben bij schaling, maar hier zijn features al op vergelijkbare schaal
            model = LinearRegression()
            param_note = "Geen hyperparameters"
        else:
            from sklearn.ensemble import RandomForestRegressor
            n_estimators = st.slider("Aantal bomen (n_estimators)", 50, 500, 200, step=50)
            max_depth = st.slider("Maximale diepte (max_depth)", 2, 20, 8, step=1)
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            param_note = f"n_estimators={n_estimators}, max_depth={max_depth}"

        test_size = st.slider("Test set grootte", 0.1, 0.5, 0.2, step=0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        colm1, colm2, colm3 = st.columns(3)
        colm1.metric("Model", model_choice)
        colm2.metric("MAE (°C)", f"{mae:.2f}")
        colm3.metric("R²", f"{r2:.3f}")
        st.caption(f"Features: {features} | {param_note}")

        # Plot: predicted vs actual
        fig_pred = px.scatter(
            x=y_test, y=y_pred,
            labels={"x": "Echte temperatuur (°C)", "y": "Voorspelde temperatuur (°C)"},
            title=f"📈 {model_choice} — Voorspelde vs. Werkelijke temperatuur"
        )
        min_ax = float(min(y_test.min(), y_pred.min()))
        max_ax = float(max(y_test.max(), y_pred.max()))
        fig_pred.add_shape(
            type="line", x0=min_ax, y0=min_ax, x1=max_ax, y1=max_ax,
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Feature importance (alleen voor RF)
        if model_choice == "Random Forest":
            importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
            fig_imp = px.bar(
                importances,
                orientation="h",
                labels={"value": "Belang", "index": "Feature"},
                title="🔎 Feature importance (Random Forest)"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("🔮 Snelle datum-voorspelling")
        st.caption("Kies een dag van het jaar. Voor extra invoer-features: gebruik de selectie bovenaan deze pagina (toonende vriendelijke namen).")

        # Enkele-datum-voorspelling op basis van seizoensfeatures (slechts één dag-slider)
        X_season = work[["sin_doy", "cos_doy", "sin_mon", "cos_mon"]]
        y_season = work["TG_C"]
        model_season = type(model)() if model_choice == "Lineaire Regressie" else type(model)(
            n_estimators=model.n_estimators if hasattr(model, "n_estimators") else 200,
            max_depth=model.max_depth if hasattr(model, "max_depth") else 8,
            random_state=42,
            n_jobs=-1 if hasattr(model, "n_jobs") else None
        )
        model_season.fit(X_season, y_season)

        sel_dayofyear = st.slider("Dag van het jaar", 1, 366, 200)

        sin_doy = math.sin(2 * math.pi * sel_dayofyear / 365.25)
        cos_doy = math.cos(2 * math.pi * sel_dayofyear / 365.25)
        # bepaal maand uit dag van het jaar (jaar 2000 is schrikkeljaar-safe voor indexering)
        from datetime import datetime, timedelta
        tmp_date = datetime(2000, 1, 1) + timedelta(days=sel_dayofyear - 1)
        sel_month = tmp_date.month
        sin_mon = math.sin(2 * math.pi * sel_month / 12.0)
        cos_mon = math.cos(2 * math.pi * sel_month / 12.0)

        temp_pred_simple = model_season.predict(
            pd.DataFrame([{"sin_doy": sin_doy, "cos_doy": cos_doy, "sin_mon": sin_mon, "cos_mon": cos_mon}])
        )[0]
        st.success(f"Geschatte etmaaltemperatuur (seizoen-only): **{temp_pred_simple:.1f} °C**")


# =========================
# === Kaart =========
# =========================
elif page == "Kaart":
    st.header("🗺️ Stations Map Vergelijking (Amsterdam • Lauwersoog • Maastricht)")

    # --- 1) Definieer de 3 stations + bronnen ---
    stations = {
        "Amsterdam": {
            "coords": (52.3702, 4.8952),
            "paths": [
                "amsterdam_2021_2022.json",
                "amsterdam_2022_2023.json",
                "amsterdam_2023_2024.json",
            ],
        },
        "Lauwersoog": {
            "coords": (53.4053, 6.2063),
            "paths": [
                "lauwersoog_2021_2022.json",
                "lauwersoog_2022_2023.json",
                "lauwersoog_2023_2024.json",
            ],
        },
        "Maastricht": {
            "coords": (50.8514, 5.6900),
            "paths": [
                "maastricht_2021_2022.json",
                "maastricht_2022_2023.json",
                "maastricht_2023_2024.json",
            ],
        },
    }   

    # --- 2) Data inlezen met je bestaande load_data() ---
    frames = []
    for city, meta in stations.items():
        parts = []
        for path in meta["paths"]:
            if not os.path.exists(path):
                st.warning(f"{city}: bestand niet gevonden → {path}")
                continue
            try:
                df_city_year = load_data(path)
                parts.append(df_city_year)
            except Exception as e:
                st.error(f"Fout bij laden {city} ({path}): {e}")
        if parts:
            df_city = pd.concat(parts, ignore_index=True)
            df_city["city"] = city
            df_city["latitude"], df_city["longitude"] = meta["coords"]
            frames.append(df_city)

    if not frames:
        st.error("Geen stationsdata beschikbaar (controleer bestandsnamen).")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True)


    # --- 3) Filters (jaar/maand en metriek) ---
    c1, c2, c3 = st.columns([1,1,2])

    years_all = sorted(df_all["year"].dropna().unique().astype(int))
    sel_years = c1.multiselect("Jaar", years_all, default=years_all)

    # kies één specifieke maand of 'Alle maanden'
    months_all = list(range(1,13))
    month_names = {i: pd.to_datetime(str(i), format="%m").month_name()[:3] for i in months_all}
    sel_month = c2.selectbox("Maand", options=["Alle"] + months_all, format_func=lambda x: x if x=="Alle" else f"{x:02d} - {month_names[x]}")

    metric_options = {
        "Gemiddelde temperatuur (°C)": ("TG_C", "mean", "°C"),
        "Minimum temperatuur (°C)"    : ("TN_C", "mean", "°C"),
        "Maximum temperatuur (°C)"    : ("TX_C", "mean", "°C"),
        "Neerslag (mm, som)"          : ("RH_mm","sum",  "mm"),
        "Zonuren (uur, som)"          : ("SQ_h", "sum",  "uur"),
        "Gemiddelde wind (m/s)"       : ("FG_ms","mean", "m/s"),
    }
    sel_metric_label = c3.selectbox("Metriek", list(metric_options.keys()))
    metric_col, agg_fn, metric_unit = metric_options[sel_metric_label]

    # --- 4) Filter toepassen ---
    dff = df_all[df_all["year"].isin(sel_years)].copy()
    if sel_month != "Alle":
        dff = dff[dff["month"] == int(sel_month)]

    # check dat metriek bestaat in alle stations
    if metric_col not in dff.columns:
        st.info(f"Metriek {metric_col} ontbreekt in data.")
        st.stop()

    # --- 5) Aggregatie per station (per jaar óf per jaar+maand) ---
    group_cols = ["city", "year"]
    if sel_month != "Alle":
        group_cols.append("month")

    agg = dff.groupby(group_cols, as_index=False)[metric_col].agg(agg_fn)

    # Voor de kaart willen we één waarde per station tonen.
    # Kies aggregatie over de geselecteerde jaren/maanden -> per station één punt:
    agg_station = dff.groupby("city", as_index=False)[metric_col].agg(agg_fn)
    # Voeg lat/lon erbij (neem de eerste – is per station constant)
    coords = dff.groupby("city")[["latitude","longitude"]].first().reset_index()
    agg_station = agg_station.merge(coords, on="city", how="left")

    # --- 6) Kaart met pydeck: kleur + radius schalen op metriek ---
    import pydeck as pdk
    val = agg_station[metric_col].fillna(0.0)
    vmin, vmax = float(val.min()), float(val.max())
    if vmin == vmax:  # voorkom divide-by-zero
        vmax = vmin + 1e-9

    def color_scale(v):
        # blauw -> geel -> rood
        # normaliseer 0..1
        t = (v - vmin) / (vmax - vmin)
        # eenvoudige 3-kleur interpolatie
        # low (0): blauw (70,120,240), mid (0.5): geel (240,200,40), high (1): rood (230,70,60)
        if t <= 0.5:
            # tussen blauw en geel
            a = t / 0.5
            r = 70 + a*(240-70)
            g = 120 + a*(200-120)
            b = 240 + a*(40-240)
        else:
            a = (t - 0.5) / 0.5
            r = 240 + a*(230-240)
            g = 200 + a*(70-200)
            b = 40  + a*(60-40)
        return [int(r), int(g), int(b), 180]

    agg_station["color"] = agg_station[metric_col].apply(color_scale)
    # radius (meters): schaal op range; caps zetten voor leesbaarheid
    rmin, rmax = 2000, 12000
    agg_station["radius"] = rmin + (agg_station[metric_col]-vmin) * (rmax-rmin) / (vmax - vmin if vmax>vmin else 1)
    agg_station[metric_col + "_txt"] = agg_station[metric_col].astype(float).map(lambda x: f"{x:.2f}")


    st.subheader("Kaart — waarde per station (kleur = laag→hoog, grootte = waarde)")
    tooltip_html = (
        "<b>{city}</b><br/>"
        + sel_metric_label + ": <b>{" + metric_col + "_txt}</b> " + metric_unit + "<br/>"
        + "Jaren: " + (", ".join(map(str, sel_years)) if sel_years else "–") + "<br/>"
        + "Maand: " + (month_names[int(sel_month)] if sel_month != "Alle" else "Alle")
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=agg_station,
        get_position=["longitude","latitude"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        stroked=True,
        get_line_color=[0,0,0,200],
        line_width_min_pixels=1,
    )

    center_lat = agg_station["latitude"].mean()
    center_lon = agg_station["longitude"].mean()
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6.2)

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": tooltip_html, "style": {"color": "white"}}
    )
    st.pydeck_chart(deck, use_container_width=True)

    # --- 7) Vergelijkingsplot (bars per station) ---
    st.subheader("Vergelijking per station")

    # Aggregatie per jaar en station
    by_year_df = dff.groupby(["year","city"], as_index=False)[metric_col].agg(agg_fn)

    if by_year_df["year"].nunique() >= 2:
        # Trend per station over de jaren (markers aan)
        figb = px.line(
            by_year_df,
            x="year", y=metric_col, color="city", markers=True,
            labels={"year": "Jaar", "city": "Station", metric_col: sel_metric_label},
            title=f"{sel_metric_label} — trend per station over jaren"
        )
        figb.update_traces(line=dict(width=2), marker=dict(size=9))
        # Optioneel: vergelijkbare y-as
        # figb.update_yaxes(rangemode="tozero")
    else:
        # Fallback: één jaar geselecteerd → dot plot per station
        y_selected = int(by_year_df["year"].iloc[0])
        dot = by_year_df.sort_values(metric_col, ascending=True)
        figb = px.scatter(
            dot, x=metric_col, y="city",
            labels={"city": "Station", metric_col: sel_metric_label},
            title=f"{sel_metric_label} — per station (jaar {y_selected})",
            color="city"
        )
        figb.update_traces(
            marker=dict(size=14, line=dict(width=1, color="rgba(0,0,0,0.3)")),
            hovertemplate="Station: %{y}<br>" + sel_metric_label + ": %{x:.2f}<extra></extra>"
        )
        figb.update_layout(xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)"))

    st.plotly_chart(figb, use_container_width=True)




    # --- 9) Kleine legenda/omschrijving ---
    st.caption(
        "Cirkels: hoe groter en roder, hoe hoger de geselecteerde metriek. "
        "Filters bovenaan: kies jaren en één maand (of Alle) om verschillen door het jaar heen te bekijken."
    )
