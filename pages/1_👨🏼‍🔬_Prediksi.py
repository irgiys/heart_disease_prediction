import pickle 
import streamlit as st

model = pickle.load(open('hypertension.sav', 'rb'))

st.set_page_config(
    page_title="Prediksi tekanan darah tinggi",
    page_icon="🩸",
)

st.markdown(
    """
        # Prediksi tekanan darah tinggi
        Untuk melakukan prediksi website ini membutuhkan **12 inputan** dengan ketentuan berdasarkan dataset sehingga menghasilkan prediksi yang lebih akurat.
    """
)
# Kategorikal data
sex_data = {0: "Perempuan", 1: "Laki-laki"}
cp_data = {0:"Tidak sakit", 1: "Angina tipikal", 2: "Angina atipikal", 3: "Non-anginal"}
yes_no_data = {0: "Tidak", 1: "Ya"}
restecg_data = {0:"Normal", 1: "ST-T Abnormal", 2: "Left ventricular"}
slope_data = {0:"Menaik", 1:"Datar", 2:"Menurun"}


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur (min=11, max=98)", min_value=11, max_value=98, placeholder=11)
with col2: 
    sex = st.selectbox("Jenis kelamin", options=list(sex_data.keys()), format_func=lambda x:sex_data[x])

with col1:
    cp = st.selectbox("Sakit dada", options=list(cp_data.keys()), format_func=lambda x:cp_data[x])
with col2: 
    tresbps = st.number_input("Tekanan darah mmHg (min=94, max=200)", min_value=94, max_value=200)

with col1:
    chol = st.number_input("Kolesterol serum mm/dl (min=126, max=564)", min_value=126, max_value=564)
with col2:
    fbs = st.selectbox("Kadar gula darah puasa > 120 mg/dL", options=list(yes_no_data.keys()), format_func=lambda x:yes_no_data[x])

with col1:
    restecg = st.selectbox("Elektrokardiografi (EKG) istirahat", options=list(restecg_data.keys()), format_func=lambda x:restecg_data[x])
with col2:
    thalach = st.number_input("Denyut jantung maksimum (min=71, max=202)", min_value=71, max_value=202)

with col1:
    exang = st.selectbox("Angina akibat olahraga", options=list(yes_no_data.keys()), format_func=lambda x:yes_no_data[x])
with col2:
    oldpeak = st.number_input("Depresi ST disebabkan oleh olahraga (min=0, max=6.2)", min_value=0.0, max_value=6.2)

slope = st.selectbox("Slope dari segmen ST yang terjadi selama tes olahraga", options=list(slope_data.keys()), format_func=lambda x:slope_data[x])
ca = st.number_input("Jumlah pembuluh darah selama prosedur flouroskopi (min=0, max=4)", min_value=0, max_value=4)


if st.button("Prediksi penyakit darah tinggi"):
    hypertension_prediction = model.predict([[age,sex,cp,tresbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca]])
    if(hypertension_prediction[0]==0):
        st.success("Pasien tidak terkena darah tinggi",icon="✅")
    else :
        st.warning("Pasien terkena darah tinggi", icon="⚠️")
        