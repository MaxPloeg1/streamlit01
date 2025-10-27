# streamlit_dashboard.py

import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
# === KPI tegels ==========
# =========================
avg_temp = df["TG_C"].mean() if "TG_C" in df.columns else np.nan
total_rain = df["RH_mm"].sum() if "RH_mm" in df.columns else np.nan
total_sun = df["SQ_h"].sum() if "SQ_h" in df.columns else np.nan

kpi1, kpi2, kpi3 = st.columns(3)
if not math.isnan(avg_temp):
    kpi1.metric("🌡️ Gemiddelde Temp (°C)", f"{avg_temp:.1f}")
if not math.isnan(total_rain):
    kpi2.metric("🌧️ Totale Neerslag (mm)", f"{total_rain:.1f}")
if not math.isnan(total_sun):
    kpi3.metric("☀️ Totale Zonuren", f"{total_sun:.1f}")

# =========================
# === Overzicht ===========
# =========================
if page == "Overzicht":
    st.header(f"🌍 {selected_city}: Het Weer in Verandering")
    st.subheader("Warmer – Droger – Zonniger (jaarvergelijking)")

    # === Data voorbereiden ===
    agg_dict = {}
    if "TG_C" in df.columns:
        agg_dict["TG_C"] = "mean"
    if "RH_mm" in df.columns:
        agg_dict["RH_mm"] = "sum"
    if "SQ_h" in df.columns:
        agg_dict["SQ_h"] = "sum"

    if not agg_dict:
        st.warning("Benodigde kolommen ontbreken voor overzicht (TG_C / RH_mm / SQ_h).")
    else:
        yearly = df.groupby("year").agg(agg_dict).reset_index()

        # Zonuren opschalen voor gecombineerde balk
        if "SQ_h" in yearly.columns:
            scale_factor = 200
            yearly["SQ_scaled"] = yearly["SQ_h"] / scale_factor
        else:
            yearly["SQ_scaled"] = 0

        # Uniforme layout
        layout_style = dict(
            font=dict(family="Arial, sans-serif", size=14, color="#ffffff"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, linecolor="grey"),
            yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center", font=dict(size=12, color="#ffffff")),
        )

        # 1) Jaarvergelijking
        fig = go.Figure()
        if "TG_C" in yearly.columns:
            fig.add_trace(go.Bar(
                x=yearly["year"], y=yearly["TG_C"],
                name="Gem. Temp (°C)", marker_color="#e74c3c",
                hovertemplate="Gem. Temp: %{y:.1f} °C<br>Jaar: %{x}<extra></extra>"
            ))

        if "SQ_h" in yearly.columns:
            fig.add_trace(go.Bar(
                x=yearly["year"], y=yearly["SQ_scaled"],
                name=f"Zonuren (x{scale_factor}h)", marker_color="#f1c40f",
                hovertemplate="Zonuren: %{customdata} uur<br>Jaar: %{x}<extra></extra>",
                customdata=yearly["SQ_h"]
            ))

        if "RH_mm" in yearly.columns:
            fig.add_trace(go.Scatter(
                x=yearly["year"], y=yearly["RH_mm"],
                name="Neerslag (mm)", mode="lines+markers",
                yaxis="y2", line=dict(color="#3498db", width=3),
                hovertemplate="Neerslag: %{y:.0f} mm<br>Jaar: %{x}<extra></extra>"
            ))

        fig.update_layout(
            title="📊 Vergelijking per jaar: Temperatuur, Neerslag en Zonuren",
            xaxis_title="Jaar",
            yaxis=dict(title="Temp (°C) & Zonuren (geschaald)", side="left"),
            yaxis2=dict(title="Neerslag (mm)", overlaying="y", side="right"),
            barmode="group"
        )
        fig.update_layout(**layout_style)
        st.plotly_chart(fig, use_container_width=True)

        # Praktische conclusie
        if len(yearly) >= 2 and "TG_C" in yearly.columns and "RH_mm" in yearly.columns:
            diff_temp = yearly["TG_C"].iloc[-1] - yearly["TG_C"].iloc[-2]
            diff_rain = yearly["RH_mm"].iloc[-2] - yearly["RH_mm"].iloc[-1]
            diff_sun = 0.0
            if "SQ_h" in yearly.columns:
                diff_sun = yearly["SQ_h"].iloc[-1] - yearly["SQ_h"].iloc[-2]
            st.info(
                f"In {int(yearly['year'].iloc[-1])} was het gemiddeld {diff_temp:.1f}°C warmer, "
                f"viel er {diff_rain:.0f} mm minder regen en scheen de zon {diff_sun:.0f} uur langer "
                f"dan in {int(yearly['year'].iloc[-2])}."
            )

        # 2) Lange termijn trend
        if "TG_C" in df.columns:
            avg_yearly_temp = df.groupby("year")["TG_C"].mean().reset_index()
            fig_trend = px.line(
                avg_yearly_temp, x="year", y="TG_C", markers=True,
                title="📈 Lange termijn trend: Gemiddelde jaartemperatuur",
                labels={"TG_C": "Gemiddelde Temp (°C)", "year": "Jaar"}
            )
            fig_trend.update_traces(line=dict(color="#e74c3c", width=4))
            fig_trend.update_layout(**layout_style)
            st.plotly_chart(fig_trend, use_container_width=True)

        # 3) Seizoensgemiddelden
        if "TG_C" in df.columns:
            season_temp = df.groupby(["year", "season"])["TG_C"].mean().reset_index()
            season_order = ["winter", "lente", "zomer", "herfst"]
            season_colors = {"winter": "#3498db", "lente": "#2ecc71", "zomer": "#f1c40f", "herfst": "#e67e22"}

            fig_season = px.bar(
                season_temp, x="year", y="TG_C", color="season",
                title="🌦️ Gemiddelde temperatuur per seizoen",
                labels={"TG_C": "Gemiddelde Temp (°C)", "year": "Jaar", "season": "Seizoen"},
                category_orders={"season": season_order},
                color_discrete_map=season_colors,
                barmode="group"
            )
            fig_season.update_layout(**layout_style)
            st.plotly_chart(fig_season, use_container_width=True)


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

        # Multiselect voor features
        default_feats = ["RH_mm", "SQ_h", "FG_ms"]
        default_feats = [f for f in default_feats if f in candidate_features]
        feature_choice = st.multiselect(
            "Kies invoer-features (naast seizoenssignal):",
            options=candidate_features,
            default=default_feats
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

        st.subheader("🔮 Snelle datum-voorspelling (alleen seizoenseffecten)")
        st.caption("Handig als je exogene variabelen (regen/zon/wind) niet weet: gebruikt enkel sin/cos van datum.")
        # Enkele-datum-voorspelling op basis van seizoensfeatures:
        # We trainen een tweede, simpel model met alleen sin/cos
        X_season = work[["sin_doy", "cos_doy", "sin_mon", "cos_mon"]]
        y_season = work["TG_C"]
        # Zelfde type model als gekozen (LR of RF), zodat consistent
        model_season = type(model)() if model_choice == "Lineaire Regressie" else type(model)(
            n_estimators=model.n_estimators if hasattr(model, "n_estimators") else 200,
            max_depth=model.max_depth if hasattr(model, "max_depth") else 8,
            random_state=42,
            n_jobs=-1 if hasattr(model, "n_jobs") else None
        )
        model_season.fit(X_season, y_season)

        dcol1, dcol2 = st.columns(2)
        sel_month = dcol1.slider("Maand", 1, 12, 7)
        sel_dayofyear = dcol2.slider("Dag van het jaar", 1, 366, 200)

        sin_doy = math.sin(2 * math.pi * sel_dayofyear / 365.25)
        cos_doy = math.cos(2 * math.pi * sel_dayofyear / 365.25)
        sin_mon = math.sin(2 * math.pi * sel_month / 12.0)
        cos_mon = math.cos(2 * math.pi * sel_month / 12.0)

        temp_pred_simple = model_season.predict(
            pd.DataFrame([{"sin_doy": sin_doy, "cos_doy": cos_doy, "sin_mon": sin_mon, "cos_mon": cos_mon}])
        )[0]
        st.success(f"Geschatte etmaaltemperatuur (alleen seizoenseffecten): **{temp_pred_simple:.1f} °C**")


# =========================
# === Kaart ===============
# =========================
elif page == "Kaart":
    st.header("🗺️ Interactieve kaart (MapLibre)")
    needed = {"latitude", "longitude", "TG_C"}
    if needed.issubset(df.columns):
        st.caption("Kleur = temperatuur (°C), grootte = neerslag (mm). Zoom en klik voor details.")

        # Zorg dat size nooit negatief is
        if "RH_mm" in df.columns:
            size_series = pd.to_numeric(df["RH_mm"], errors="coerce").fillna(0).clip(lower=0)
        else:
            size_series = None

        size_max = st.slider("Maximale bubbelgrootte (px)", 5, 50, 20, step=1)

        fig_map = px.scatter_map(
            df,
            lat="latitude",
            lon="longitude",
            color="TG_C",
            size=size_series,        # array-like toegestaan
            size_max=size_max,       # >>> juiste manier om bubbelgrootte te begrenzen
            hover_name=df["date"].dt.strftime("%Y-%m-%d"),
            hover_data={
                "TG_C": True,
                "RH_mm": "RH_mm" in df.columns,
                "SQ_h": "SQ_h" in df.columns,
                "latitude": False,
                "longitude": False,
            },
            title=f"Meetpunten — {selected_city}",
            color_continuous_scale="RdYlBu_r",
            zoom=6,
            height=600,
            map_style="carto-positron",
        )

        st.plotly_chart(fig_map, use_container_width=True)
    else:
        ontbrekend = ", ".join(sorted(needed - set(df.columns)))
        st.info(f"Geen locatiegegevens beschikbaar (mis: {ontbrekend}). "
                f"Coördinaten per stad worden automatisch gezet; controleer of de dataset geladen is.")
