import pickle 
# import numpy as np
import streamlit as st

model = pickle.load(open('hypertension.sav', 'rb'))
st.title("Prediksi darah tinggi")

col1, col2, col3 = st.columns(3)
with col1 : 
    age = st.number_input("umur")
with col2 :
    sex = st.number_input("Jenis kelamin")
with col3 : 
    cp = st.number_input("cp")
with col1 : 
    tresbps = st.number_input("tresbps")
with col2 :
    chol = st.number_input("chol")
with col3 : 
    fbs = st.number_input("fbs")
with col1 : 
    restecg = st.number_input("restecg")
with col2 :
    thalach = st.number_input("thalach")
with col3 : 
    exang = st.number_input("exang")
with col1 : 
    oldpeak = st.number_input("oldpeak")
with col2 :
    slope = st.number_input("slope")
with col3 : 
    ca = st.number_input("ca")
with col1 : 
    thal = st.number_input("thal")

diagnosis = ''
if st.button("Prediksi penyakit darah tinggi"):
    hypertension_prediction = model.predict([[age,sex,cp,tresbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if(hypertension_prediction[0]==0):
        diagnosis = "Pasien tidak terkena darah tinggi"
    else :
        diagnosis = "Pasien terkena penyakit darah tinggii"
st.success(diagnosis)