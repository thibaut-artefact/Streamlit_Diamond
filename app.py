import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import numpy as np
import pandas as pd
import requests
import random
from typing import Any, Dict, Optional
import time

# --- Page configuration ----------------------------------------------------
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
)

# --- Styling & Typography --------------------------------------------------
st.markdown(
    """
    <style>
        /* Import luxury fonts from Google */
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Montserrat:wght@300;400;500;600&display=swap');

        /* Apply Typography */
        html, body, [class*="st-"] {
            font-family: 'Montserrat', sans-serif;
        }
        h1, h2, h3, .stHeader, .stSubheader {
            font-family: 'Cinzel', serif !important;
            letter-spacing: 1px;
        }

        /* Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #0b1620 0%, #1b2a3b 45%, #243d5a 100%);
            color: #f0f8ff;
        }

        /* --- REMOVE TOP WHITE BAR --- */
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }

        /* Ensure all standard text is high-contrast */
        .stMarkdown, .stText, .stHeader, .stSubheader, .stMetric {
            color: #f7fbff !important;
        }
        .stMetric * {
            color: #f7fbff !important;
        }

        /* --- LEFT PANEL (SIDEBAR) FIXES --- */
        [data-testid="stSidebar"] {
            background-color: rgba(11, 22, 32, 0.4) !important;
            backdrop-filter: blur(15px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        
        /* Force all sidebar text, widget labels, and headers to be white */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] div {
            color: #f7fbff !important;
        }
        /* Keep selectbox dropdown text dark so it's readable on the white dropdown background */
        div[role="listbox"] span {
            color: #0b1620 !important;
        }

        /* --- RIGHT PANEL FIXES --- */
        /* Glassmorphism Metric Cards */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
        }

        /* Metric Delta Visibility */
        [data-testid="stMetricDelta"] {
            background-color: rgba(255, 255, 255, 0.08) !important;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
        }
        [data-testid="stMetricDelta"] > div {
            color: #ffffff !important;
            font-weight: 500;
        }

        /* Animated Button */
        .stButton>button {
            background: linear-gradient(90deg, #ff7e5f, #feb47b);
            color: #0b1620;
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            border: none;
            padding: 0.65rem 1.2rem;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 32px rgba(0, 0, 0, 0.35);
        }

        /* Custom Spinning Diamond Loader */
        @keyframes spin-3d {
            0% { transform: rotateY(0deg) scale(1); }
            50% { transform: rotateY(180deg) scale(1.1); }
            100% { transform: rotateY(360deg) scale(1); }
        }
        .custom-spinner {
            font-size: 3.5rem;
            display: inline-block;
            animation: spin-3d 1.5s infinite ease-in-out;
            text-shadow: 0 0 24px rgba(255, 255, 255, 0.9);
            margin: 20px auto;
            text-align: center;
            width: 100%;
        }
        .spinner-text {
            text-align: center;
            font-family: 'Cinzel', serif;
            color: #ffd700;
            font-size: 1.2rem;
            letter-spacing: 2px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helpers ---------------------------------------------------------------

API_URL = "https://diamonds-861302064365.europe-west1.run.app/predict_one"

@st.cache_data(show_spinner=False)
def get_prediction(diamond_data: Dict[str, Any]) -> Optional[float]:
    try:
        response = requests.post(API_URL, json=diamond_data, timeout=10)
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            if "prediction" in payload:
                return float(payload["prediction"])
            if "price" in payload:
                return float(payload["price"])

        return float(payload)
    except (requests.RequestException, ValueError, TypeError):
        return None

def make_curve_data(
    base_data: Dict[str, Any],
    base_carat: float,
    span: float = 0.5,
    points: int = 11,
) -> pd.DataFrame:
    low = max(0.1, base_carat - span)
    high = base_carat + span
    carats = np.linspace(low, high, points)

    rows = []
    for c in carats:
        point = dict(base_data)
        point["carat"] = float(round(c, 3))
        price = get_prediction(point)
        rows.append({"carat": c, "price": price})

    return pd.DataFrame(rows)

def format_currency(value: float) -> str:
    return f"${value:,.2f}"

# --- UI --------------------------------------------------------------------

st.markdown("## 💎 Diamond Price Predictor")
st.markdown("Configure your jewel in the sidebar to receive an instant market appraisal.")

with st.sidebar:
    st.header("The 4 Cs")
    carat = st.slider("Carat", 0.1, 6.0, 0.80, 0.01, help="The physical weight of the diamond. One carat is equal to 0.20 grams. Larger diamonds are rarer and price increases exponentially with carat weight.")
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"], help="Grades from D (colorless) to Z (light yellow). D, E, and F are considered 'colorless' and are the most valuable.")
    clarity = st.selectbox("Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2"], help="Measures internal inclusions. IF (Internally Flawless) is the highest, followed by VVS (Very, Very Slightly Included).")
    cut = st.selectbox("Cut", ["Ideal", "Premium", "Good", "Very Good", "Fair"], help="How well proportioned the diamond is. The cut impacts how light travels through the stone. 'Ideal' reflects the most light.")

    st.subheader("Dimensions")
    depth = st.slider("Depth (%)", 30, 100, 60, 1, help="Total depth percentage.")
    table = st.slider("Table (%)", 30, 100, 55, 1, help="Width of the diamond's top facet relative to its widest point.")
    x = st.slider("Length (mm)", 0.0, 15.0, 5.0, 0.1)
    y = st.slider("Width (mm)", 0.0, 60.0, 30.0, 0.1)
    z = st.slider("Depth (mm)", 0.0, 60.0, 30.0, 0.1)

    st.markdown("---")
    submit = st.button("Appraise Diamond")

# --- Prediction & Animation Logic ---
if submit:
    input_data = {
        "carat": carat,
        "color": color,
        "clarity": clarity,
        "cut": cut,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
    }

    loading_placeholder = st.empty()
    loading_placeholder.markdown(
        """
        <div class="custom-spinner">💎</div>
        <div class="spinner-text">Appraising your diamond...</div>
        """, 
        unsafe_allow_html=True
    )

    predicted_price = get_prediction(input_data)
    
    baseline_data = input_data.copy()
    baseline_data["carat"] = 1.0
    baseline_price = get_prediction(baseline_data)

    loading_placeholder.empty()

    if predicted_price is None:
        st.error("Could not fetch a prediction from the API. Please try again or check your network.")
    else:
        price_per_carat = predicted_price / carat if carat else None

        js_code = """
        <script>
            const doc = window.parent.document;
            const container = doc.createElement('div');
            container.style.position = 'fixed';
            container.style.top = '0';
            container.style.left = '0';
            container.style.width = '100vw';
            container.style.height = '100vh';
            container.style.pointerEvents = 'none';
            container.style.zIndex = '999999';
            container.style.overflow = 'hidden';
            doc.body.appendChild(container);

            const style = doc.createElement('style');
            style.innerHTML = `
                @keyframes diamond-stack {
                    0% { transform: translateY(-10vh) rotate(0deg); }
                    100% { transform: translateY(var(--stop-y)) rotate(var(--rot)); }
                }
                .custom-diamond {
                    position: absolute;
                    top: -10%;
                    animation: diamond-stack var(--fall-dur) ease-in forwards;
                    text-shadow: 0 0 16px rgba(255, 255, 255, 0.9);
                    opacity: 0; 
                }
                @keyframes fade-in {
                    to { opacity: 1; }
                }
            `;
            doc.head.appendChild(style);

            for (let i = 0; i < 100; i++) {
                const diamond = doc.createElement('div');
                diamond.innerHTML = '💎';
                diamond.className = 'custom-diamond';
                
                diamond.style.left = (Math.random() * 96) + 'vw';
                diamond.style.setProperty('--fall-dur', (Math.random() * 2 + 1.5) + 's');
                
                const delay = (Math.random() * 11) + 's';
                diamond.style.animationDelay = delay + ', ' + delay;
                diamond.style.animationName = 'diamond-stack, fade-in';
                diamond.style.animationDuration = 'var(--fall-dur), 0.2s';
                
                diamond.style.setProperty('--stop-y', (Math.random() * 8 + 88) + 'vh');
                diamond.style.setProperty('--rot', (Math.random() * 720 - 360) + 'deg');
                diamond.style.fontSize = (Math.random() * 1.5 + 1.2) + 'rem';
                
                container.appendChild(diamond);
            }

            setTimeout(() => {
                container.style.transition = 'opacity 1s ease-out';
                container.style.opacity = '0';
                
                setTimeout(() => {
                    container.remove();
                    style.remove();
                }, 1000);
            }, 14000); 
        </script>
        """
        components.html(js_code, height=0, width=0)

        curve_df = make_curve_data(input_data, base_carat=carat, span=max(0.15, carat * 0.3), points=9)

        left, right = st.columns([2, 1])

        with left:
            st.subheader("Valuation Curve")
            chart = (
                alt.Chart(curve_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("carat:Q", title="Carat"),
                    y=alt.Y("price:Q", title="Predicted Price (USD)"),
                    color=alt.condition(
                        alt.datum.carat == float(carat),
                        alt.value("#ffd700"),
                        alt.value("#7fc8ff"),
                    ),
                    opacity=alt.condition(
                        alt.datum.carat == float(carat),
                        alt.value(1.0),
                        alt.value(0.55),
                    ),
                    tooltip=[
                        alt.Tooltip("carat:Q", title="Carat", format=".2f"),
                        alt.Tooltip("price:Q", title="Predicted price", format="$,.2f"),
                    ],
                )
                .properties(height=420)
                .interactive()
            )

            highlight = (
                alt.Chart(pd.DataFrame([{"carat": carat, "price": predicted_price}]))
                .mark_circle(size=140, color="#ffdd33")
                .encode(
                    x="carat:Q",
                    y="price:Q",
                    tooltip=[
                        alt.Tooltip("carat:Q", title="Selected carat", format=".2f"),
                        alt.Tooltip("price:Q", title="Predicted price", format="$,.2f"),
                    ],
                )
            )

            st.altair_chart(chart + highlight, use_container_width=True)
            st.markdown("*The curve estimates how price changes when carat varies while keeping other attributes constant.*")

        with right:
            st.subheader("Appraisal Summary")
            
            if baseline_price is not None:
                delta_val = predicted_price - baseline_price
                sign = "+" if delta_val >= 0 else "-"
                delta_str = f"{sign} ${abs(delta_val):,.2f} vs 1.0ct equivalent"
            else:
                delta_str = None

            st.metric(
                label="Estimated Market Value", 
                value=format_currency(predicted_price),
                delta=delta_str,
                help="The delta compares your diamond to a 1.0 carat diamond with the exact same color, cut, and clarity."
            )
            
            if price_per_carat is not None:
                st.metric(label="Price per Carat", value=format_currency(price_per_carat))

            st.markdown("---")
            st.markdown("**Expert Tip:** Notice how stepping up from a 0.99ct to a 1.00ct diamond causes a significant jump in price due to 'magic sizes' in the diamond industry.")