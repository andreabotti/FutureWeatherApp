"""
Welcome page content: bilingual copy and single entry point for the app.
"""
import streamlit as st

COPY = {
    "EN": {
        "title": "Welcome to the App",
        "tagline": "Explore future climate scenarios for building performance simulation.",
        "getting_started_title": "How to use this app",
        "step1": "Choose **Future Weather Scenarios** to compare current (TMYx) with projected climate (e.g. RCP 4.5 2050, RCP 8.5 2080).",
        "step2": "Use **Current Weather Data (TMYx / CTI)** to inspect typical-year hourly data and charts by location.",
        "step3": "Open **IPCC Scenarios** (under Future Weather Scenarios) for global temperature context and scenario definitions.",
        "tmyx_title": "What is TMYx?",
        "tmyx_body": (
            "**TMYx (Typical Meteorological Year Extended)** is a representative year of weather data "
            "used as a baseline for building energy simulation. It is built from multiple years of observations "
            "so it reflects typical conditions rather than a single extreme year."
        ),
        "ipcc_title": "IPCC scenarios (RCP / SSP)",
        "ipcc_body": (
            "**RCP** (Representative Concentration Pathway) and **SSP** (Shared Socioeconomic Pathway) are "
            "standardized greenhouse-gas pathways used by the IPCC. They are **scenarios**, not predictions: "
            "they describe *if* society follows a certain path, *then* climate may respond in a certain way. "
            "RCP 4.5 and RCP 8.5 are commonly used for building simulation (intermediate and high emissions)."
        ),
    },
    "IT": {
        "title": "Benvenuto nell'app",
        "tagline": "Esplora gli scenari climatici futuri per la simulazione delle prestazioni degli edifici.",
        "getting_started_title": "Come usare questa app",
        "step1": "Scegli **Scenari climatici futuri** per confrontare i dati attuali (TMYx) con il clima previsto (es. RCP 4.5 2050, RCP 8.5 2080).",
        "step2": "Usa **Dati meteo attuali (TMYx / CTI)** per esaminare i dati orari dell'anno tipo e i grafici per località.",
        "step3": "Apri **Scenari IPCC** (sotto Scenari climatici futuri) per il contesto sulle temperature globali e le definizioni degli scenari.",
        "tmyx_title": "Cos'è il TMYx?",
        "tmyx_body": (
            "**TMYx (Typical Meteorological Year Extended)** è un anno rappresentativo di dati meteorologici "
            "usato come baseline per la simulazione energetica degli edifici. È costruito da più anni di osservazioni "
            "quindi riflette condizioni tipiche anziché un singolo anno estremo."
        ),
        "ipcc_title": "Scenari IPCC (RCP / SSP)",
        "ipcc_body": (
            "**RCP** (Representative Concentration Pathway) e **SSP** (Shared Socioeconomic Pathway) sono "
            "percorsi standardizzati di gas serra usati dall'IPCC. Sono **scenari**, non previsioni: "
            "descrivono *se* la società segue un certo percorso, *allora* il clima può rispondere in un certo modo. "
            "RCP 4.5 e RCP 8.5 sono spesso usati per la simulazione degli edifici (emissioni intermedie e alte)."
        ),
    },
}


def render_welcome_page() -> None:
    """Render the Welcome tab: title, tagline, 3 columns (getting started, TMYx, IPCC). Language from st.session_state['ui_lang']."""
    lang = st.session_state.get("ui_lang", "EN")
    c = COPY.get(lang, COPY["EN"])

    st.markdown(f"##### {c['title']}")
    st.markdown(c["tagline"])
    st.markdown("")

    col1, col2, col3 = st.columns([3,2,2], gap="medium")

    with col1:
        st.markdown(f"##### {c['getting_started_title']}")
        st.markdown(f"1. {c['step1']}")
        st.markdown(f"2. {c['step2']}")
        st.markdown(f"3. {c['step3']}")

    with col2:
        st.markdown(f"##### {c['tmyx_title']}")
        st.markdown(c["tmyx_body"])

    with col3:
        st.markdown(f"##### {c['ipcc_title']}")
        st.markdown(c["ipcc_body"])
