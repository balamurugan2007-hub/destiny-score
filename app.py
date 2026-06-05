import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Model load பண்றோம்
model = pickle.load(open('model/destiny_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Destiny Score", page_icon="🔮", layout="centered")

st.title("🔮 Destiny Score Predictor")
st.markdown("### உன் daily habits சொல்லு — உன் future பாப்போம்!")
st.markdown("---")

# User Input
col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("😴 Sleep Hours", 4.0, 9.0, 7.0)
    exercise = st.slider("🏃 Exercise Days/Week", 0, 7, 3)
    junk_food = st.slider("🍔 Junk Food Days/Week", 0, 7, 2)
    stress = st.slider("😰 Stress Level (1-10)", 1, 10, 5)
    water = st.slider("💧 Water Intake (Litres)", 1.0, 4.0, 2.5)

with col2:
    screen_time = st.slider("📱 Screen Time (hrs)", 1.0, 12.0, 4.0)
    social_media = st.slider("📲 Social Media (hrs)", 0.0, 6.0, 2.0)
    study = st.slider("📚 Study Hours/Day", 0.0, 8.0, 3.0)
    savings = st.slider("💰 Savings %", 0.0, 50.0, 20.0)
    reading = st.slider("📖 Reading Hours/Day", 0.0, 3.0, 0.5)

st.markdown("---")

if st.button("🔮 Predict My Destiny Score!"):
    # Prediction
    input_data = np.array([[sleep, exercise, junk_food, screen_time,
                            social_media, study, stress, savings,
                            reading, water]])
    input_scaled = scaler.transform(input_data)
    score = model.predict(input_scaled)[0]
    score = round(float(score), 2)

    # Score display
    st.markdown(f"## 🎯 உன் Destiny Score: **{score} / 100**")

    # Grade
    if score >= 80:
        grade = "🌟 Excellent — உன் future bright-ஆ இருக்கு!"
        color = "green"
    elif score >= 60:
        grade = "👍 Good — சின்னதா improve பண்ணா top-ல போவ!"
        color = "blue"
    elif score >= 40:
        grade = "⚠️ Average — இப்பவே habits change பண்ணு!"
        color = "orange"
    else:
        grade = "🚨 Low — உடனே lifestyle மாத்தணும்!"
        color = "red"

    st.markdown(f"### :{color}[{grade}]")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Destiny Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#ff4b4b"},
                {'range': [40, 60], 'color': "#ffa500"},
                {'range': [60, 80], 'color': "#4b9eff"},
                {'range': [80, 100], 'color': "#00cc44"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Tips
    st.markdown("### 💡 Improvement Tips:")
    if sleep < 7:
        st.warning("😴 தூக்கம் 7+ hours வேணும்!")
    if exercise < 3:
        st.warning("🏃 Week-ல 3+ days exercise பண்ணு!")
    if stress > 7:
        st.warning("😰 Stress குறைக்க meditation try பண்ணு!")
    if social_media > 3:
        st.warning("📱 Social media time குறைக்கணும்!")
    if study < 2:
        st.warning("📚 Daily 2+ hours படிக்கணும்!")
    if savings < 10:
        st.warning("💰 குறைஞ்சது 10% savings வச்சுக்கோ!")