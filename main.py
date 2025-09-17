# === Imports & theme ===
import json, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")

# === Data inlezen (robust voor list-of-records of {"data": [...]}) ===
PATH = "amsterdam_2023_2024.json"
with open(PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
df = pd.json_normalize(records)

# === Datum parsen ===
# Probeer automatisch, anders val terug op KNMI-formaat (YYYYMMDD)
try:
    df["date"] = pd.to_datetime(df["date"])
except Exception:
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

# === Hulpfunctie: maak kolom als die bestaat, gedeeld door 10 (KNMI tienden) ===
def make_scaled(src, dest, divisor=10):
    if src in df.columns:
        df[dest] = pd.to_numeric(df[src], errors="coerce") / divisor

# === Veelgebruikte grootheden (als aanwezig) ===
# Temperatuur (°C)
make_scaled("TG", "TG_C")   # daggem. temp
make_scaled("TN", "TN_C")   # dag-min
make_scaled("TX", "TX_C")   # dag-max
make_scaled("T10N", "T10N_C")  # 10cm-min (optioneel)

# Wind (m/s)
make_scaled("FG", "FG_ms")    # gem. windsnelheid
make_scaled("FXX", "FXX_ms")  # hoogste windstoot
# Windrichting (graden) zit meestal in DDVEC (al in graden)
if "DDVEC" in df.columns:
    df["DDVEC"] = pd.to_numeric(df["DDVEC"], errors="coerce")

# Neerslag (mm) & verdamping (mm)
make_scaled("RH", "RH_mm")     # neerslagsom
make_scaled("EV24", "EV24_mm") # verdamping

# Zonduur (uren) — KNMI 'SQ' is in tienden van uren
make_scaled("SQ", "SQ_h")      

# Luchtdruk (hPa)
make_scaled("PG", "PG_hPa")  # daggemiddelde druk
make_scaled("PX", "PX_hPa")  # dag-maximum
make_scaled("PN", "PN_hPa")  # dag-minimum

# === Tijdsfeatures ===
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["weekday"] = df["date"].dt.weekday  # 0=maandag

# Seizoen (NL)
def season(m):
    return ("winter" if m in [12,1,2] else
            "lente"  if m in [3,4,5] else
            "zomer"  if m in [6,7,8] else
            "herfst")
df["season"] = df["month"].apply(season)

df.head()
