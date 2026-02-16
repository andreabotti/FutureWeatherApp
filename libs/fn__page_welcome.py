"""
Welcome page content: bilingual copy and single entry point for the app.
"""
import streamlit as st

COPY = {
    "EN": {
        "title": "Welcome",
        "tagline": (
            "Compare recent climate and future scenarios to understand how summer conditions and overheating risk "
            "may change for buildings."
        ),
        "left_title": "Why this app",
        "left_body": (
            "In design practice we often rely on “reference” climate inputs used for checks and comparisons. "
            "The issue is that the climate is changing: real summers are already hotter, and future projections suggest "
            "further warming.\n\n"
            "This app helps you compare recent and future climate conditions to support decisions on comfort, passive strategies, "
            "and HVAC approach."
        ),
        "norms_title": "Standard data vs real-world climate",
        "norms_body": (
            "For compliance checks (Italy’s Minimum Requirements technical report), “official” climate data is required. "
            "In Italy, this reference is provided by the UNI 10349 series. "
            "It supports consistency and compliance, but it may not fully represent recent heatwaves and more extreme summer conditions."
        ),
        "interpret_title": "How to interpret",
        "interpret_body": (
            "**TMYx (baseline):** “today / recent”  \n"
            "**TMYx → rcp45_2050:** same location, plausible 2050  \n"
            "**TMYx → rcp85_2080:** more severe 2080"
        ),
        "tmyx_title": "What is TMYx",
        "tmyx_body": (
            "TMYx is a “typical” weather file: it does not represent a single year, but a representative year built from multiple years of data. "
            "It is useful for energy simulations and consistent comparisons between locations."
        ),
        "ipcc_title": "IPCC scenarios (RCP / SSP)",
        "ipcc_body": (
            "RCP and SSP are scenario frameworks used by the IPCC to describe plausible emissions and climate pathways. "
            "They are not exact forecasts: they are “what if” scenarios. "
            "In this app we use them to explore how climate could change in the future (e.g., 2050, 2080) and test design resilience."
        ),
    },
    "IT": {
        # NOTE: requested removal of the IT title/tagline block.
        "left_title": "Perché questa app",
        "left_body": (
            "Nella pratica progettuale usiamo spesso dati climatici di riferimento per verifiche e simulazioni. "
            "Il problema è che il clima sta cambiando: le estati reali sono già più calde e le proiezioni future "
            "indicano un ulteriore aumento.\n\n"
            "Questa app permette di confrontare clima recente e clima futuro per supportare scelte progettuali "
            "su comfort estivo, strategie passive e impostazione impiantistica."
        ),
        "norms_title": "Dati normativi e dati reali",
        "norms_body": (
            "Per le verifiche energetiche (Relazione tecnica dei Requisiti Minimi) servono dati climatici ufficiali. "
            "In Italia questi dati sono forniti dalla serie UNI 10349. "
            "Sono utili per la conformità normativa, ma possono non rappresentare pienamente le ondate di calore recenti "
            "e le condizioni estive più estreme."
        ),
        "interpret_title": "Come interpretare",
        "interpret_body": (
            "**TMYx (baseline):** “oggi / recente”  \n"
            "**TMYx → rcp45_2050:** stesso luogo, possibile futuro 2050  \n"
            "**TMYx → rcp85_2080:** futuro più severo 2080"
        ),
        "tmyx_title": "Cos’è TMYx",
        "tmyx_body": (
            "TMYx è un file climatico tipico: non descrive un singolo anno, ma un anno rappresentativo "
            "costruito a partire da più anni di dati. È utile per simulazioni energetiche coerenti."
        ),
        "ipcc_title": "Scenari IPCC (RCP / SSP)",
        "ipcc_body": (
            "I Percorsi Rappresentativi di Concentrazione (Representative Concentration Pathways, RCP) sono scenari climatici "
            "espressi in termini di concentrazioni di gas serra piuttosto che in termini di livelli di emissioni. "
            "Il numero associato a ciascun RCP si riferisce al Forzante Radiativo (Radiative Forcing – RF), espresso in W/m², "
            "e indica l’entità dei cambiamenti climatici antropogenici entro il 2100 rispetto al periodo preindustriale.\n\n"
            "In pratica, ciascun RCP mostra una diversa quantità di calore addizionale immagazzinato nel sistema Terra "
            "come risultato delle emissioni di gas serra.\n\n"
            "**In questa app si considerano in particolare:**\n\n"
            "- **RCP8.5** (\"Business-as-usual\" / nessuna mitigazione): crescita delle emissioni ai ritmi attuali. "
            "Entro il 2100 le concentrazioni atmosferiche di CO₂ arrivano a circa 840–1120 ppm, cioè 3–4 volte i livelli "
            "preindustriali (280 ppm).\n"
            "- **RCP4.5** (\"Forte mitigazione\"): prevede l’attuazione di politiche di controllo delle emissioni. "
            "È uno scenario di stabilizzazione: entro il 2070 le emissioni di CO₂ scendono al di sotto dei livelli attuali "
            "e la concentrazione atmosferica si stabilizza, entro fine secolo, a circa il doppio dei livelli preindustriali."
        ),
    },
}


def render_welcome_page() -> None:
    """
    Render the Welcome tab (bilingual).
    Language from st.session_state['ui_lang'].
    """
    lang = st.session_state.get("ui_lang", "EN")
    c = COPY.get(lang, COPY["EN"])

    # Title/tagline are optional (removed for IT by request).
    title = c.get("title")
    tagline = c.get("tagline")
    if title:
        st.markdown(f"##### {title}")
    if tagline:
        st.markdown(tagline)
    if title or tagline:
        st.markdown("")

    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"##### {c['left_title']}")
        st.markdown(c["left_body"])
        st.markdown("")
        st.markdown(f"##### {c['norms_title']}")
        st.markdown(c["norms_body"])
        st.markdown("")
        st.markdown(f"##### {c['interpret_title']}")
        st.markdown(c["interpret_body"])

    with col2:
        tab_tmyx, tab_ipcc = st.tabs([c["tmyx_title"], c["ipcc_title"]])
        with tab_tmyx:
            st.markdown(c["tmyx_body"])
        with tab_ipcc:
            st.markdown(c["ipcc_body"])
