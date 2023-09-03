import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import altair as alt
import plotly.express as px
import pickle as pkl

st.set_page_config(page_title="LOS Prediction",page_icon="⚕️",layout="wide",initial_sidebar_state="expanded")

@st.cache_data
def data():
    data = pd.read_csv("./data/los.csv.gz", compression='gzip', error_bad_lines=False)
    return data

@st.cache_data
def admin():
    admin = pd.read_csv("./data/admissions.csv.gz", compression='gzip', error_bad_lines=False)
    return admin

@st.cache_data
def patient():
    patient = pd.read_csv("./data/patients.csv.gz", compression='gzip', error_bad_lines=False)
    return patient

def disease():
    disease = pd.read_csv("./data/disease.csv.gz", compression='gzip', error_bad_lines=False)
    return disease

a = data()
b = admin()
c = patient()
d = disease()

def prediction_los(input_data):
    load_model = pkl.load(open('./data/los_model.sav', 'rb'))

    input_data_as_numpy_array = np.asarray(input_data)
    # Convert the tuple to a 2D numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Now you can use the model to predict
    predict_los = load_model.predict(input_data_reshaped)
    predictions = predict_los[0]
    st.header("Predicted LOS:")
    st.write(predictions, 'days')

def los_prediction():
    html_temp = """ 
        <div style ="padding:13px;padding-bottom:50px"> 
        <h1 style ="color:white;text-align:center;">Length of Stay Prediction</h1> 
        </div> 
        """
    st.markdown(html_temp, unsafe_allow_html = True) 
    st.write("---")
    st.header("PLEASE INPUT THE MEDICAL RECORD")
    input_data = list()
    col1,col2,col3 = st.columns(3)

    with col1:
        anion_gap = st.text_input("Anion Gap Level")
        co2 = st.text_input("Calculated Total CO2 Level")
        free_calcium = st.text_input('Free Calcium Level')
        hematocrit = st.text_input('Hematocrit Level')
        i = st.text_input('I Level')
        lactate = st.text_input('Lactate Level')
        mcv = st.text_input('Mean Corpuscular Volume Level')
        pt = st.text_input('Prothrombin Time')
        phosphate = st.text_input('Phosphate Level')
        potassium_WB = st.text_input('Potassium Whole Blood Level')
        rapamycin = st.text_input('Rapamycin Level')
        urea_nitrogen = st.text_input('Urea Nitrogen Level')
        pH = st.text_input('pH Level')
        temp_gender = st.radio('Gender',
                          ("Male",
                           "Female"
                          ))
        if temp_gender == 'Male':
            gender = '0'
        else:
            gender = '1'
        diastolic_blood_pressure = st.text_input('Arterial Blood Pressure Diastolic Level')
        eye_opening = st.text_input('GCS - Eye Opening')
        heart_rate = st.text_input('Heart Rate')
        nibp_mean = st.text_input('Non Invasive Blood Pressure Mean Level')
        resp_rate = st.text_input('Respiratory Rate')

    with col2:
        bicarbonate = st.text_input("Bicarbonate Level")
        chloride = st.text_input("Chloride Level")
        glucose = st.text_input('Glucose Level')
        hemoglobin = st.text_input('Hemoglobin Level')
        inr = st.text_input('INR(P/T) Level')
        mch = st.text_input('Mean Corpuscular Hemoglobin Level')
        magnesium = st.text_input('Magnesium Level')
        ptt = st.text_input('Partial Thromboplastin Time')
        platelet_count = st.text_input('Platelet Count Level')
        rdw = st.text_input('Red Cell Distribution Width') 
        red_blood_cell = st.text_input('Red Blood Cells Level')
        white_blood_cell = st.text_input('White Blood Cells Level')
        po2 = st.text_input('PO2 Level')
        age = st.text_input('Age')
        arterial_blood_pressure_mean = st.text_input('Arterial Blood Pressure Mean Level')
        motor_response = st.text_input('GCS - Motor Response')
        o2_fraction = st.text_input('Inspired O2 Fraction Level')
        nibp_systolic = st.text_input('Non Invasive Blood Pressure Systolic Level')
        temperature = st.text_input('Temperature (°F)')
        
    with col3:        
        calcium = st.text_input('Total Calcium Level')
        creatinine = st.text_input('Creatinine Level')
        h = st.text_input('H Level')
        heparin = st.text_input('Heparin Level')
        l = st.text_input('L Level')
        mchc = st.text_input('Mean Corpuscular Hemoglobin Concentration Level')
        oxy_saturation = st.text_input('Oxygen Saturation Level')
        phenobarbital = st.text_input('Phenobarbital Level')
        potassium = st.text_input('Potassium Level')
        rdw_sd = st.text_input('Red Cell Distribution Width - Standard Deviation Level')
        sodium = st.text_input('Sodium Level')
        pco2 = st.text_input('pCO2 Level')
        tacroFK = st.text_input('Tacrolimus FK Level')
        jh_hlm = st.text_input('Activity / Mobility (JH-HLM)')
        systolic_blood_pressure = st.text_input('Arterial Blood Pressure Systolic Level')
        verbal_response = st.text_input('GCS - Verbal Response')
        nibp_diastolic = st.text_input('Non Invasive Blood Pressure Diastolic Level')
        o2_saturation = st.text_input('O2 Saturation Pulseoxymetry Level')
        
    if st.button('LOS Result'):
        input_data.extend([anion_gap, bicarbonate, calcium, co2, chloride, creatinine, free_calcium, glucose, h, hematocrit, hemoglobin, heparin, i, inr, l,
                      lactate, mch, mchc, mcv, magnesium, oxy_saturation, pt, ptt, phenobarbital, phosphate, platelet_count, potassium, potassium_WB, rdw,
                      rdw_sd, rapamycin, red_blood_cell, sodium, urea_nitrogen, white_blood_cell, pco2, pH, po2, tacroFK, gender, age, jh_hlm, diastolic_blood_pressure,
                      arterial_blood_pressure_mean, systolic_blood_pressure, eye_opening, motor_response, verbal_response, heart_rate, o2_fraction, nibp_diastolic, nibp_mean,
                      nibp_systolic, o2_saturation, resp_rate, temperature])
        

        numeric_input_data = [float(value) if value and value != "0" else 0 for value in input_data]
        prediction_los(numeric_input_data)

def los_visualization_menu(a,b,c):
        html_temp = """ 
        <div style ="padding:13px;padding-bottom:50px"> 
        <h1 style ="color:white;text-align:center;">Length of Stay Visualization</h1> 
        </div> 
        """
        st.markdown(html_temp, unsafe_allow_html = True) 
        menu = st.radio("SELECT THE VISUALIZATION TYPE",
                        ("Length of Stay",
                        "MIMIC-IV"
                        )
        )
        
        if menu == 'MIMIC-IV':
            los_visualization(a,b,c)
        else:
            los_filter(a)


def los_visualization(a,b,c):
    st.write('\n')
    st.text('Loading data...')
    data = a
    admin = b
    patient = c

    col1, col2 = st.columns(2)
    with col1:
        #ROW 1 - LOS HOSPITAL ADMINSSIONS
        st.subheader('Distribution of LOS for all Hospital Admissions')
        fig1 = px.histogram(data, x='los', nbins=200, range_x=[0, 50],
                        labels={'los': 'Length of Stay (days)', 'count': 'Count'},
                        )
        fig1.update_traces(marker=dict(line=dict(color='black', width=1)))
        st.plotly_chart(fig1)

        #ROW 2 - GENDER
        # Replace 0 with 'Male' and 1 with 'Female' in the 'gender' column
        data['gender'].replace({0: 'Male', 1: 'Female'}, inplace=True)
        gender_counts = data['gender'].value_counts()
        st.subheader('Gender')
        st.bar_chart(gender_counts)
        
    with col2:
        #ROW 1 - AGE
        st.subheader('Age')
        fig2 = px.histogram(patient, x='anchor_age', nbins=20,
                        labels={'anchor_age': 'Age', 'count': 'Count'},
                        )
        fig2.update_traces(marker=dict(line=dict(color='black', width=1)))
        st.plotly_chart(fig2)

        #ROW 2 - RACE
        admin['race'].replace(regex=r'^ASIAN\D*', value='ASIAN', inplace=True)
        admin['race'].replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)
        admin['race'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)
        admin['race'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)
        admin['race'].replace(['UNABLE TO OBTAIN', 'OTHER', 'PATIENT DECLINED TO ANSWER', 
                                'UNKNOWN/NOT SPECIFIED'], value='OTHER/UNKNOWN', inplace=True)
        admin['race'].loc[~admin['race'].isin(admin['race'].value_counts().nlargest(5).index.tolist())] = 'OTHER/UNKNOWN'
        admin['race'].value_counts()

        st.subheader('Races of Patients')
        fig4 = px.histogram(admin, y='race', nbins=200,
                        labels={'race': 'Race', 'count': 'Count'},
                        )
        st.plotly_chart(fig4)
    
def los_filter(a):
        st.write('\n')
        st.text('Loading data...')
        data = a

        data['gender'].replace({0: 'MALE', 1: 'FEMALE'}, inplace=True)
        gender = st.sidebar.multiselect("Select the Gender:",
                                options=data["gender"].unique(),
                                default=data["gender"].unique()
                                )
        if not gender:
            st.warning("Please select at least one gender.")
            return
        age = st.sidebar.number_input("Age:", min_value=18, max_value=100, step=1, format="%d")
        
        df_selection = data.query(
            "gender == @gender &"
            "anchor_age == @age"
        )

        st.title(":bar_chart: Length of Stay Dashboard")
        st.markdown("##")

        lowest_stay = int(df_selection["los"].min())
        highest_stay = int(df_selection["los"].max())
        average_stay = round(df_selection["los"].mean(),1)

        col1, col2,col3 = st.columns(3)
        with col1:
            st.subheader("Lowest Length of Stay")
            st.subheader(f"{lowest_stay}")
        with col2:
            st.subheader("Highest Length of Stay")
            st.subheader(f"{highest_stay}")
        with col3:
            st.subheader("Aevrage Length of Stay")
            st.subheader(f"{average_stay}")
        st.markdown("---")

        st.subheader('Distribution of LOS for all Hospital Admissions')
        fig1 = px.histogram(df_selection, x='los', nbins=200, range_x=[0, 115],
                        labels={'los': 'Length of Stay (days)', 'count': 'Count'},
                        )
        fig1.update_traces(marker=dict(line=dict(color='black', width=1)))
        fig1.update_layout(width=800, height=600)
        st.plotly_chart(fig1, use_container_width=True)

def disease_prediction():
    html_temp = """ 
        <div style ="padding:13px;padding-bottom:50px"> 
        <h1 style ="color:white;text-align:center;">Disease Prediction</h1> 
        </div> 
        """
    st.markdown(html_temp, unsafe_allow_html = True) 
    st.write("---")
    st.header("PLEASE INPUT THE MEDICAL RECORD")
    input_data = list()
    col1,col2,col3 = st.columns(3)

    with col1:
        anion_gap = st.text_input("Anion Gap Level")
        co2 = st.text_input("Calculated Total CO2 Level")
        free_calcium = st.text_input('Free Calcium Level')
        hematocrit = st.text_input('Hematocrit Level')
        i = st.text_input('I Level')
        lactate = st.text_input('Lactate Level')
        mcv = st.text_input('Mean Corpuscular Volume Level')
        pt = st.text_input('Prothrombin Time')
        phosphate = st.text_input('Phosphate Level')
        potassium_WB = st.text_input('Potassium Whole Blood Level')
        rapamycin = st.text_input('Rapamycin Level')
        urea_nitrogen = st.text_input('Urea Nitrogen Level')
        pH = st.text_input('pH Level')
        jh_hlm = st.text_input('Activity / Mobility (JH-HLM)')
        systolic_blood_pressure = st.text_input('Arterial Blood Pressure Systolic Level')
        verbal_response = st.text_input('GCS - Verbal Response')
        nibp_diastolic = st.text_input('Non Invasive Blood Pressure Diastolic Level')
        o2_saturation = st.text_input('O2 Saturation Pulseoxymetry Level')
        age = st.text_input('Age')

    with col2:
        bicarbonate = st.text_input("Bicarbonate Level")
        chloride = st.text_input("Chloride Level")
        glucose = st.text_input('Glucose Level')
        hemoglobin = st.text_input('Hemoglobin Level')
        inr = st.text_input('INR(P/T) Level')
        mch = st.text_input('Mean Corpuscular Hemoglobin Level')
        magnesium = st.text_input('Magnesium Level')
        ptt = st.text_input('Partial Thromboplastin Time')
        platelet_count = st.text_input('Platelet Count Level')
        rdw = st.text_input('Red Cell Distribution Width') 
        red_blood_cell = st.text_input('Red Blood Cells Level')
        white_blood_cell = st.text_input('White Blood Cells Level')
        po2 = st.text_input('PO2 Level')
        diastolic_blood_pressure = st.text_input('Arterial Blood Pressure Diastolic Level')
        eye_opening = st.text_input('GCS - Eye Opening')
        heart_rate = st.text_input('Heart Rate')
        nibp_mean = st.text_input('Non Invasive Blood Pressure Mean Level')
        resp_rate = st.text_input('Respiratory Rate')
        temp_gender = st.radio('Gender',
                    ("Male",
                    "Female"
                    ))
        if temp_gender == 'Male':
            gender = '0'
        else:
            gender = '1'

    with col3:
        calcium = st.text_input('Total Calcium Level')
        creatinine = st.text_input('Creatinine Level')
        h = st.text_input('H Level')
        heparin = st.text_input('Heparin Level')
        l = st.text_input('L Level')
        mchc = st.text_input('Mean Corpuscular Hemoglobin Concentration Level')
        oxy_saturation = st.text_input('Oxygen Saturation Level')
        phenobarbital = st.text_input('Phenobarbital Level')
        potassium = st.text_input('Potassium Level')
        rdw_sd = st.text_input('Red Cell Distribution Width - Standard Deviation Level')
        sodium = st.text_input('Sodium Level')
        pco2 = st.text_input('pCO2 Level')
        tacroFK = st.text_input('Tacrolimus FK Level')
        arterial_blood_pressure_mean = st.text_input('Arterial Blood Pressure Mean Level')
        motor_response = st.text_input('GCS - Motor Response')
        o2_fraction = st.text_input('Inspired O2 Fraction Level')
        nibp_systolic = st.text_input('Non Invasive Blood Pressure Systolic Level')
        temperature = st.text_input('Temperature (°F)')

    if st.button('Disease Result'):
        input_data.extend([anion_gap, bicarbonate, calcium, co2, chloride, creatinine, free_calcium, glucose, h, hematocrit, hemoglobin, heparin, i, inr, l,
                        lactate, mch, mchc, mcv, magnesium, oxy_saturation, pt, ptt, phenobarbital, phosphate, platelet_count, potassium, potassium_WB, rdw,
                        rdw_sd, rapamycin, red_blood_cell, sodium, urea_nitrogen, white_blood_cell, pco2, pH, po2, tacroFK, jh_hlm, diastolic_blood_pressure,
                        arterial_blood_pressure_mean, systolic_blood_pressure, eye_opening, motor_response, verbal_response, heart_rate, o2_fraction, nibp_diastolic, nibp_mean,
                        nibp_systolic, o2_saturation, resp_rate, temperature, age, gender])
    

    numeric_input_data = [float(value) if value and value != "0" else 0 for value in input_data]
    prediction_disease(numeric_input_data)

def prediction_disease(input_data):
    load_model = pkl.load(open('./data/disease_model.sav', 'rb'))

    column_names = ["blood", "circulatory", "congenital", "digestive", "endocrine", "genitourinary", "infectious", "injury", "mental", "misc", "muscular", "neoplasms", "nervous", "pregnancy", "prenatal", "respiratory", "skin"]

    input_data_as_numpy_array = np.asarray(input_data)

    # Convert the tuple to a 2D numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Use the model to predict
    predict_los = load_model.predict(input_data_reshaped)
    # Ensure predictions are in dense array format
    dense_predictions = predict_los.toarray()[0]
    
    predicted_df = pd.DataFrame([dense_predictions], columns=column_names)

    # Display predicted labels in a table format
    st.header("Predicted Labels:")
    st.dataframe(predicted_df)


def disease_visual(d):
    html_temp = """ 
            <div style ="padding:13px;padding-bottom:50px"> 
            <h1 style ="color:white;text-align:center;">Disease Visualization</h1> 
            </div> 
            """
    st.markdown(html_temp, unsafe_allow_html = True) 
    disease = d

    choice = st.sidebar.radio("Select the Disease: ",
                              ("BLOOD",
                               "CIRCULATORY",
                               "CONGENITAL",
                               "DIGESTIVE",
                               "ENDOCRINE",
                               "GENITOURINARY",
                               "INFECTIOUS",
                               "INJURY",
                               "MENTAL",
                               "MISC",
                               "MUSCULAR",
                               "NEOPLASMS",
                               "NERVOUS",
                               "PREGNANCY",
                               "PRENATAL",
                               "RESPIRATORY",
                               "SKIN"))
    

    choice = choice.lower()

    st.title(":bar_chart: Disease Dashboard")
    st.write('\n')
    st.text('Loading data...')

    male = (disease[(disease[choice.lower()] == 1) & (disease['gender'] == 0)]).shape[0]
    female = (disease[(disease[choice.lower()] == 1) & (disease['gender'] == 1)]).shape[0]
    col1, col2= st.columns(2)
    with col1:
        st.subheader("Male")
        st.subheader(male)

    with col2:
        st.subheader("Female")
        st.subheader(female)
        
    st.markdown("---")

    st.subheader('NUMBER OF PEOPLE HAS "' + choice.upper() + '" DISEASE')
    fig = px.histogram(disease[disease[choice] == 1], x='age', nbins=200, range_x=[0,100],
                    labels={'age':'Age', 'count':'Count'}
                    )
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)

page = st.sidebar.selectbox('SELECT PAGE',['LOS-Predictions','LOS-Visualization', 'Disease-Predictions', 'Disease-Visualization']) 
st.sidebar.write("---")
if page == 'LOS-Predictions':
    los_prediction()
elif page == 'Disease-Predictions':
    disease_prediction()
elif page == 'LOS-Visualization':
    los_visualization_menu(a,b,c)
else:
    disease_visual(d)
