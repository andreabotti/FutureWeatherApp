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
        "map_view": "Map View",
        "table_view": "Table View",
        "current_weather_data_tmyx": "Current Weather Data (TMYx / CTI)",
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
        "map_view": "Mappa",
        "table_view": "Tabella",
        "current_weather_data_tmyx": "Temperature correnti (TMYx)",
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
