# Bilingual UI labels: EN / IT
# Use label(key) or label(key, lang) for current or explicit language.

LABELS = {
    "EN": {
        "baseline_climate_file": "Baseline Climate File",
        "current_climate_baseline": "Current climate (baseline)",
        "climate_scenarios": "Climate Scenarios",
        "year": "Year",
        "metric": "Metric",
        "welcome": "Welcome",
        "future_weather_scenarios": "Future Weather Scenarios",
        "future_temperatures_italy": "Future Temperatures",
        "future_vs_current_temperatures_italy": "Future vs Current Temperatures",
        "future_climate_italian_regions": "Future Climate — Italian Regions",
        "future_climate_italian_regions_style_ref": "Confronto Regione — Style Reference",
        "map_view": "Map View",
        "table_view": "Table View",
        "current_weather_data_tmyx": "Typical Meteorological Files (TMY)",
        "tmyx_tab_intro": (
            "This tab shows the **Typical Meteorological Year (TMY)** data for each weather station. "
            "Each TMY file is a synthetic year built from multiple years of observed data and represents "
            "typical (not extreme) conditions. Where multiple TMYx time-window variants are available "
            "(e.g. 2004–2018, 2007–2021, 2009–2023), they reflect different recent periods and can be "
            "compared here to understand how typical conditions have shifted over time. "
            "These files are used as baseline inputs for the climate morphing process that generates future scenarios."
        ),
        "ipcc_scenarios": "IPCC Scenarios",
    },
    "IT": {
        "baseline_climate_file": "Scenario climatico di base",
        "current_climate_baseline": "Clima attuale (baseline)",
        "climate_scenarios": "Scenari climatici",
        "year": "Anno",
        "metric": "Metrica",
        "welcome": "Benvenuti",
        "future_weather_scenarios": "Scenari Climatici Futuri",
        "future_temperatures_italy": "Proiezioni Temperature Future",
        "future_vs_current_temperatures_italy": "Temperature Correnti vs Future",
        "future_climate_italian_regions": "Confronto Regione per Regione",
        "future_climate_italian_regions_style_ref": "Confronto Regione — Riferimento stili",
        "map_view": "Mappa",
        "table_view": "Tabella",
        "current_weather_data_tmyx": "File Meteo Tipici (TMY)",
        "tmyx_tab_intro": (
            "Questo tab mostra i dati **TMY (Typical Meteorological Year)** per ogni stazione meteorologica. "
            "Ogni file TMY è un anno sintetico costruito a partire da più anni di dati osservati e rappresenta "
            "condizioni tipiche (non estreme). Quando sono disponibili più varianti TMYx su finestre temporali "
            "diverse (es. 2004–2018, 2007–2021, 2009–2023), ciascuna riflette un periodo recente differente "
            "e può essere confrontata qui per capire come le condizioni tipiche sono cambiate nel tempo. "
            "Questi file sono utilizzati come input di base per il processo di morfing climatico che genera gli scenari futuri."
        ),
        "ipcc_scenarios": "Scenari IPCC",
    },
}


def label(key: str, lang: str | None = None) -> str:
    """Return the label for `key` in the current or given language (default: EN)."""
    if lang is None:
        try:
            import streamlit as st
            lang = st.session_state.get("ui_lang", "EN")
        except Exception:
            lang = "EN"
    return LABELS.get(lang, LABELS["EN"]).get(key, key)
