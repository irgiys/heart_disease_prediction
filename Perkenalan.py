import streamlit as st

st.set_page_config(
    page_title="Perkenalan website",
    page_icon="🫀",
)
st.write("# 🫀 Perkenalan Website")

st.markdown(
    """
    Website ini memungkinkan  untuk memprediksi sakit jantung dengan menggunakan dataset yang berasal dari Kaggle yaitu data hasil dari **70,692 respon survei BRFSS 2015**. Dengan menggunakan algoritma `Logistic Regression` prediksi model ini mencapai score akurasi **86.08845%**
    # Resource
    - Dataset [Diabetes, Hypertension and Stroke Prediction](https://www.kaggle.com/datasets/prosperchuks/health-dataset?select=hypertension_data.csv) 
    - >  Abaikan nama file hipertensi karena sebenarnya bukan data hipertensi tetapi sakit jantung
    - Repositori [irgiys/heart_disease_predicion](https://github.com/irgiys/heart_disease_prediction)
"""
)
