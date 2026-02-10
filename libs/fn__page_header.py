# IMPORT LIBRARIES
import streamlit as st
from PIL import Image
from .fn__libs import *


def f001__create_page_header():

    ##### PAGE CONFIG
    st.set_page_config(page_title="EETRA Future Weather App", page_icon='./img/EETRA_favicon.png', layout="wide")

    # Inject fonts and styles
    st.markdown("""
        <link href="https://fonts.cdnfonts.com/css/ronzino" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Lora:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            .block-container {
                padding-top: 0.5rem;
                padding-bottom: 0.5rem;
                padding-left: 2rem;
                padding-right: 2rem;
                font-family: 'Ronzino', 'Inter', sans-serif;
            }
            h2.custom-title {
                font-family: 'Lora', 'Source Serif Pro', 'Source Serif 4', serif;
                font-weight: 400;
                color: #2a7d2e;
                margin-bottom: -20px;
            }
            .custom-caption {
                font-size: 0.8rem;
                color: gray;
            }
            /* Compact App Language radio: less space below label */
            .stRadio label {
                margin-bottom: 0.15rem !important;
            }
            .stRadio [role="radiogroup"] {
                margin-top: 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    ##### TOP CONTAINER
    top_col1, spacing, top_col2, lang_col = st.columns([20, 0.1, 200, 40])

    with lang_col:
        st.session_state.setdefault("ui_lang", "EN")
        lang_choice = st.radio(
            "App Language",
            options=["🇬🇧 ENG", "🇮🇹 ITA"],
            index=1 if st.session_state["ui_lang"] == "IT" else 0,
            key="header_lang_radio",
            horizontal=True,
            label_visibility="visible",
        )
        st.session_state["ui_lang"] = "IT" if "ITA" in lang_choice else "EN"

    with top_col2:
        st.write('')
        st.markdown(
            """
            <div style="margin: -18px 0px;">
                <h2 class="custom-title">Future Weather App</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with top_col1:
        try:
            image = Image.open("./img/EETRA_logo_rect.png")

            from io import BytesIO
            import base64
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()

            st.markdown(
                f"""
                <div style="margin: 22px 0px;">
                    <img src="data:image/png;base64,{img_str}" style="width:100%; height:auto;" />
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.warning(f"Logo could not be loaded: {e}")



    # SVG divider (tiled horizontal greca)
    # st.markdown(
    #     """
    #     <div style="
    #         background-image: url('https://eetra.it/wp-content/uploads/2023/06/greca-verdone-thin.svg');
    #         background-repeat: repeat-x;
    #         background-position: left;
    #         background-size: auto 10px;
    #         height: 10px;
    #         margin: 0rem 0px 1.5rem 0px;">
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    # custom_hr()


f041__create_page_header = f001__create_page_header  # alias for backward compatibility
