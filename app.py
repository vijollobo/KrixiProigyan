import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from translations import LANGUAGES, t, local_crop_name, CROP_INFO, local_number, format_number
from crop_media import show_crop_photo
from model import load_data, train_model, crop_nutrient_stats, predict_top_n, FEATURES
from agri_logic import (
    soil_health_index,
    soil_health_label,
    irrigation_advice,
    fertilizer_advice,
    disease_risk_alerts,
    yield_outlook,
)

st.set_page_config(page_title="KrixiProigyan", page_icon="🌾", layout="wide")

INDIAN_STATES = [
    "Assam", "West Bengal", "Punjab", "Uttar Pradesh", "Maharashtra",
    "Karnataka", "Tamil Nadu", "Bihar", "Odisha", "Madhya Pradesh",
    "Rajasthan", "Gujarat", "Kerala", "Andhra Pradesh", "Telangana", "Goa",
]


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model(df):
    return train_model(df)


@st.cache_data
def get_crop_stats(df):
    return crop_nutrient_stats(df)


df = get_data()
model_bundle = get_model(df)
model = model_bundle["model"]
crop_stats = get_crop_stats(df)

# ---------------- Sidebar ----------------
lang_display = st.sidebar.selectbox("🌐 Language", list(LANGUAGES.keys()))
lang = LANGUAGES[lang_display]

st.sidebar.markdown(f"### {t(lang, 'sidebar_header')}")
region = st.sidebar.selectbox(t(lang, "region_label"), INDIAN_STATES)

input_mode = st.sidebar.radio(
    t(lang, "input_mode_label"),
    [t(lang, "input_mode_manual"), t(lang, "input_mode_simulate")],
)

if "sensor_values" not in st.session_state:
    sample = df.sample(1, random_state=None).iloc[0]
    st.session_state.sensor_values = {f: float(sample[f]) for f in FEATURES}

sidebar_field_labels = {
    "N": t(lang, "nitrogen"), "P": t(lang, "phosphorus"), "K": t(lang, "potassium"),
    "temperature": t(lang, "temperature"), "humidity": t(lang, "humidity"),
    "ph": t(lang, "ph"), "rainfall": t(lang, "rainfall"),
}

if input_mode == t(lang, "input_mode_simulate"):
    if st.sidebar.button("🔄 " + t(lang, "simulate_button")):
        sample = df.sample(1).iloc[0]
        st.session_state.sensor_values = {f: float(sample[f]) for f in FEATURES}
    vals = st.session_state.sensor_values
    for f in FEATURES:
        st.sidebar.metric(sidebar_field_labels[f], format_number(lang, vals[f], 2))
else:
    vals = {}
    vals["N"] = st.sidebar.slider(t(lang, "nitrogen"), 0, 140, int(st.session_state.sensor_values["N"]))
    vals["P"] = st.sidebar.slider(t(lang, "phosphorus"), 5, 145, int(st.session_state.sensor_values["P"]))
    vals["K"] = st.sidebar.slider(t(lang, "potassium"), 5, 205, int(st.session_state.sensor_values["K"]))
    vals["temperature"] = st.sidebar.slider(t(lang, "temperature"), 8.0, 45.0, float(st.session_state.sensor_values["temperature"]))
    vals["humidity"] = st.sidebar.slider(t(lang, "humidity"), 10.0, 100.0, float(st.session_state.sensor_values["humidity"]))
    vals["ph"] = st.sidebar.slider(t(lang, "ph"), 3.0, 10.0, float(st.session_state.sensor_values["ph"]))
    vals["rainfall"] = st.sidebar.slider(t(lang, "rainfall"), 15.0, 300.0, float(st.session_state.sensor_values["rainfall"]))
    st.session_state.sensor_values = vals

    # Sliders are native browser widgets and always render Western digits —
    # that's a Streamlit/browser constraint we can't override. This summary
    # line shows the same values in the selected language's native script.
    localized_summary = "  |  ".join(
        f"{sidebar_field_labels[key]}: {format_number(lang, vals[key], 2)}"
        for key in FEATURES
    )
    st.sidebar.caption(localized_summary)

# ---------------- Header ----------------
st.title("🌾 " + t(lang, "app_title"))
st.caption(t(lang, "app_subtitle") + f"  |  📍 {region}")

# ---------------- Core computations ----------------
top3 = predict_top_n(model, vals, n=3)
top_crop = top3[0][0]
top_conf = top3[0][1]

shi = soil_health_index(vals["N"], vals["P"], vals["K"], vals["ph"], vals["humidity"])
shi_label = soil_health_label(shi)
irrigation_status = irrigation_advice(vals["humidity"], vals["rainfall"], vals["temperature"])
fert_status = fertilizer_advice(vals["N"], vals["P"], vals["K"], crop_stats[top_crop])
alerts = disease_risk_alerts(vals["temperature"], vals["humidity"], vals["ph"])
yield_status = yield_outlook(shi, irrigation_status)

# ---------------- Tabs ----------------
tab_crop, tab_soil, tab_fert, tab_advisory, tab_model = st.tabs([
    t(lang, "tab_crop"), t(lang, "tab_soil"), t(lang, "tab_fert"),
    t(lang, "tab_advisory"), t(lang, "tab_model"),
])

# ---- Crop Recommendation ----
with tab_crop:
    top_crop_local_name = local_crop_name(lang, top_crop)
    col_photo, col1, col2 = st.columns([0.9, 1, 1.3])

    with col_photo:
        wiki_title = CROP_INFO.get(top_crop, {}).get("wiki", top_crop.title())
        show_crop_photo(st, top_crop, wiki_title, top_crop_local_name)

    with col1:
        st.metric(t(lang, "recommended_crop"), top_crop_local_name)
        st.metric(t(lang, "confidence"), format_number(lang, top_conf * 100, 2) + "%")
        st.markdown(f"**{t(lang, 'yield_title')}**")
        st.markdown(f"##### {t(lang, f'yield_{yield_status}')}")

    with col2:
        st.subheader(t(lang, "top3_title"))
        crops_names = [local_crop_name(lang, c) for c, _ in top3]
        confs = [p * 100 for _, p in top3]
        conf_text = [format_number(lang, c, 2) + "%" for c in confs]
        fig = px.bar(
            x=confs, y=crops_names, orientation="h",
            labels={"x": "Confidence (%)", "y": ""},
            color=confs, color_continuous_scale="Greens",
            text=conf_text,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=280)
        st.plotly_chart(fig, use_container_width=True)

# ---- Soil Health & Irrigation ----
with tab_soil:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t(lang, "soil_health_index"))
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=shi,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2C5F2D"},
                "steps": [
                    {"range": [0, 45], "color": "#F9E795"},
                    {"range": [45, 70], "color": "#CADCFC"},
                    {"range": [70, 100], "color": "#97BC62"},
                ],
            },
        ))
        fig.update_layout(height=230, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h4 style='text-align:center'>{format_number(lang, shi, 2)} / {local_number(lang, '100')}</h4>", unsafe_allow_html=True)
        st.info(t(lang, f"soil_health_{shi_label}"))
    with col2:
        st.subheader(t(lang, "irrigation_advice_title"))
        icon = {"now": "🚿", "soon": "⏳", "not_needed": "✅"}[irrigation_status]
        st.markdown(f"## {icon} {t(lang, 'irrigate_' + irrigation_status)}")
        st.write(
            f"**{t(lang, 'rainfall')}:** {format_number(lang, vals['rainfall'], 2)} mm  |  "
            f"**{t(lang, 'humidity')}:** {format_number(lang, vals['humidity'], 2)}%"
        )

# ---- Fertilizer Advisory ----
with tab_fert:
    st.subheader(t(lang, "fert_advice_title") + f" — {top_crop_local_name}")
    cols = st.columns(3)
    nutrient_labels = {"N": t(lang, "nitrogen"), "P": t(lang, "phosphorus"), "K": t(lang, "potassium")}
    status_color = {"low": "🟡", "ok": "🟢", "high": "🔴"}
    for i, nutrient in enumerate(["N", "P", "K"]):
        with cols[i]:
            status = fert_status[nutrient]
            st.metric(nutrient_labels[nutrient], format_number(lang, vals[nutrient], 2))
            st.markdown(f"{status_color[status]} **{t(lang, 'fert_' + status)}**")

    mean_n, _ = crop_stats[top_crop]["N"]
    mean_p, _ = crop_stats[top_crop]["P"]
    mean_k, _ = crop_stats[top_crop]["K"]
    comp_df = pd.DataFrame({
        "Nutrient": ["N", "N", "P", "P", "K", "K"],
        "Type": ["Current", "Ideal (avg)"] * 3,
        "Value": [vals["N"], mean_n, vals["P"], mean_p, vals["K"], mean_k],
    })
    comp_df["Label"] = comp_df["Value"].apply(lambda v: format_number(lang, v, 2))
    fig = px.bar(comp_df, x="Nutrient", y="Value", color="Type", barmode="group",
                 color_discrete_sequence=["#B85042", "#97BC62"], text="Label")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

# ---- Farmer Advisory (multilingual) ----
with tab_advisory:
    st.subheader(t(lang, "advisory_title"))

    st.markdown(f"#### {t(lang, 'alerts_title')}")
    if alerts:
        for a in alerts:
            st.warning(t(lang, "alert_" + a))
    else:
        st.success(t(lang, "no_alerts"))

    st.markdown(f"#### 📱 {t(lang, 'advisory_sms')}")
    sms_text = (
        f"{t(lang, 'recommended_crop')}: {top_crop_local_name} "
        f"({format_number(lang, top_conf * 100, 2)}%). "
        f"{t(lang, 'irrigation_advice_title')}: {t(lang, 'irrigate_' + irrigation_status)}. "
        f"{t(lang, 'soil_health_index')}: {format_number(lang, shi, 2)}/{local_number(lang, '100')} "
        f"({t(lang, 'soil_health_' + shi_label)})."
    )
    st.code(sms_text, language=None)

    st.markdown(f"#### 🔊 {t(lang, 'advisory_voice')}")
    voice_text = (
        f"{t(lang, 'recommended_crop')} — {top_crop_local_name}. "
        f"{t(lang, 'irrigate_' + irrigation_status)}. "
        + (t(lang, 'alert_' + alerts[0]) if alerts else t(lang, 'no_alerts'))
    )
    st.text_area(" ", voice_text, height=100, label_visibility="collapsed")
    st.caption("🔈 Text-to-speech playback would run through a regional-language voice model in production; shown here as script text.")

# ---- Model Insights ----
with tab_model:
    col1, col2 = st.columns(2)
    with col1:
        st.metric(t(lang, "model_accuracy"), format_number(lang, model_bundle["accuracy"] * 100, 2) + "%")
        st.subheader(t(lang, "feature_importance"))
        fi = model_bundle["feature_importances"]
        fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())}).sort_values("Importance")
        fi_df["Label"] = fi_df["Importance"].apply(lambda v: format_number(lang, v, 2))
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=["#065A82"], text="Label")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader(t(lang, "confusion_matrix"))
        cm = model_bundle["confusion_matrix"]
        labels = model_bundle["labels"]
        local_labels = [local_crop_name(lang, c) for c in labels]
        fig = px.imshow(cm, x=local_labels, y=local_labels, color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(t(lang, "footer_note"))
