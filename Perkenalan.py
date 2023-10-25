
import streamlit as st

st.set_page_config(
    page_title="Perkenalan website",
    page_icon="🩸",
)
st.write("# Prediksi tekanan darah tinggi")

st.markdown(
    """
    Dalam era digital, memiliki alat yang dapat membantu memantau kesehatan sangat penting. Website ini memungkinkan  untuk memprediksi tingkat tekanan darah tinggi dengan tingkat akurasi **81.197256%**. 
    Website ini menggunakan dataset yang berasal dari Kaggle yaitu data hasil dari **70,692 respon survei BRFSS 2015**
    # Resource
    - Dataset [Diabetes, Hypertension and Stroke Prediction](https://www.kaggle.com/datasets/prosperchuks/health-dataset?select=hypertension_data.csv)
    - Repositori [irgiys/hypertension_predicion](https://github.com/irgiys/hypertension_prediction)
"""
)
