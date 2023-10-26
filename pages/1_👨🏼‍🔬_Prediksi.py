import pickle 
import streamlit as st

model = pickle.load(open('heart_disease.sav', 'rb'))

st.set_page_config(
    page_title="Prediksi Sakit Jantung",
    page_icon="🫀",
)

st.markdown(
    """
        # 👨‍⚕️ Prediksi Sakit Jantung
        Untuk melakukan prediksi website ini membutuhkan **13 inputan** dengan ketentuan tertentu sehingga menghasilkan prediksi yang lebih akurat.
    """
)
# Kategorikal data
sex_data = {0: "Perempuan", 1: "Laki-laki"}
cp_data = {0:"Asymptotic", 1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal"}
yes_no_data = {0: "Tidak", 1: "Ya"}
restecg_data = {0:"Normal", 1: "ST-T Abnormal", 2: "Left ventricular"}
slope_data = {0:"Upsloping", 1:"Flat", 2:"Downsloping"}
thal_data = {1:"Normal", 2:"Fixed defect", 3:"Reversable"}


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
thal = st.selectbox("Thalamesia", options=list(thal_data.keys()), format_func=lambda x:thal_data[x])


if st.button("Prediksi penyakit sakit jantung"):
    heart_disease_predict = model.predict([[age,sex,cp,tresbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if(heart_disease_predict[0]==0):
        st.success("Pasien tidak terkena sakit jantung",icon="✅")
    else :
        st.warning("Pasien terkena sakit jantung", icon="⚠️")
        