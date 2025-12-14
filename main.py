import streamlit as st
import pandas as pd
import numpy as np
import joblib

try:
    model = joblib.load('gradient_boosting_regressor_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error(
        "خطأ: لم يتم العثور على ملفات النموذج. تأكد من وجود 'gradient_boosting_regressor_model.joblib' و 'scaler.joblib' في نفس المجلد.")
    st.stop()
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل الملفات: {e}")
    st.stop()

gender_options = ['Female', 'Male', 'Non-Binary']
business_travel_options = ['Frequent Travel', 'No Travel', 'Some Travel']
department_options = ['Human Resources', 'Sales', 'Technology']
education_field_options = [
    'Computer Science', 'Economics', 'Human Resources', 'Life Sciences',
    'Marketing', 'Medical', 'Other', 'Technical Degree'
]
job_role_options = [
    'Business Analyst', 'Data Scientist', 'Engineering Manager', 'HR Business Partner',
    'HR Generalist', 'Product Manager', 'Recruiter', 'Sales Executive', 'Software Engineer','Sales Representative'
]
marital_status_options = ['Divorced', 'Married', 'Single']
over_time_options = ['No', 'Yes']

le_mappings = {
    'Gender': {val: i for i, val in enumerate(sorted(gender_options))},
    'BusinessTravel': {val: i for i, val in enumerate(sorted(business_travel_options))},
    'Department': {val: i for i, val in enumerate(sorted(department_options))},
    'EducationField': {val: i for i, val in enumerate(sorted(education_field_options))},
    'JobRole': {val: i for i, val in enumerate(sorted(job_role_options))},
    'MaritalStatus': {val: i for i, val in enumerate(sorted(marital_status_options))},
    'OverTime': {val: i for i, val in enumerate(sorted(over_time_options))},
}

max_values = {
    'Age': 60.0,
    'DistanceFromHome (KM)': 45.0,  # من الرسم البياني (cell 8)
    'Salary': 550000.0,  # من الرسم البياني (cell 8)
    'JobRole': 12.0,  # من الرسم البياني (cell 11) - هذه النقطة مهمة
    'YearsAtCompany': 10.0,  # من الرسم البياني (cell 8) - *** هذه هي القيمة المستهدفة ***
    'YearsInMostRecentRole': 10.0,  # من الرسم البياني (cell 8)
    'YearsSinceLastPromotion': 10.0,  # من الرسم البياني (cell 8)
    'YearsWithCurrManager': 10.0  # من الرسم البياني (cell 8)
}

FEATURE_NAMES = [
    'Gender', 'Age', 'BusinessTravel', 'Department', 'DistanceFromHome (KM)',
    'Education', 'EducationField', 'JobRole', 'MaritalStatus', 'Salary',
    'StockOptionLevel', 'OverTime', 'YearsInMostRecentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]
# streamlit
st.set_page_config(layout="wide", page_title="Employee Tenure Predictor")
st.title('Employee Service Duration Prediction🧑‍💼')
st.write("يستخدم هذا التطبيق النموذج الذي قمت بتدريبه (Gradient Boosting) للتنبؤ بـ 'YearsAtCompany'.")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Personal Information")
    age = st.number_input(' (Age)', min_value=18, max_value=int(max_values['Age']), value=30)
    gender = st.selectbox(' (Gender)', options=gender_options)
    marital_status = st.selectbox(' (MaritalStatus)', options=marital_status_options)
    distance_from_home = st.number_input('DistanceFromHome (KM)', min_value=1,
                                         max_value=int(max_values['DistanceFromHome (KM)']), value=10)

with col2:
    st.header("Job Details")
    job_role = st.selectbox(' (JobRole)', options=job_role_options)
    department = st.selectbox(' (Department)', options=department_options)
    education_field = st.selectbox(' (EducationField)', options=education_field_options)
    education = st.slider(' (EducationLevel)', min_value=1, max_value=5, value=3)

with col3:
    st.header("Work-Related Inputs")
    salary = st.number_input(' (Salary)', min_value=10000, max_value=int(max_values['Salary']), value=50000,
                             step=1000)
    business_travel = st.selectbox(' (BusinessTravel)', options=business_travel_options)
    over_time = st.selectbox(' (OverTime)', options=over_time_options)
    stock_option_level = st.slider(' (StockOptionLevel)', min_value=0, max_value=3, value=0)
    years_in_most_recent_role = st.number_input(' (YearsInMostRecentRole)', min_value=0,
                                                max_value=int(max_values['YearsInMostRecentRole']), value=2)
    years_since_last_promotion = st.number_input(' (YearsSinceLastPromotion)', min_value=0,
                                                 max_value=int(max_values['YearsSinceLastPromotion']), value=1)
    years_with_curr_manager = st.number_input(' (YearsWithCurrManager)', min_value=0,
                                              max_value=int(max_values['YearsWithCurrManager']), value=3)

# --- 4. زر التنبؤ والمنطق ---
if st.button('predict', use_container_width=True):

    # 4a. تجميع المدخلات وترميزها (Label Encoding - مثل الخلية 12)
    data = {
        'Gender': le_mappings['Gender'][gender],
        'Age': age,
        'BusinessTravel': le_mappings['BusinessTravel'][business_travel],
        'Department': le_mappings['Department'][department],
        'DistanceFromHome (KM)': distance_from_home,
        'Education': education,
        'EducationField': le_mappings['EducationField'][education_field],
        'JobRole': le_mappings['JobRole'][job_role],
        'MaritalStatus': le_mappings['MaritalStatus'][marital_status],
        'Salary': salary,
        'StockOptionLevel': stock_option_level,
        'OverTime': le_mappings['OverTime'][over_time],
        'YearsInMostRecentRole': years_in_most_recent_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }

    # Normalization
    data['Age'] /= max_values['Age']
    data['DistanceFromHome (KM)'] /= max_values['DistanceFromHome (KM)']
    data['Salary'] /= max_values['Salary']
    data['JobRole'] /= max_values['JobRole']  # تحجيم الـ JobRole بعد ترميزه
    data['YearsInMostRecentRole'] /= max_values['YearsInMostRecentRole']
    data['YearsSinceLastPromotion'] /= max_values['YearsSinceLastPromotion']
    data['YearsWithCurrManager'] /= max_values['YearsWithCurrManager']

    # 4c. تجميع المدخلات في DataFrame بالترتيب الصحيح
    try:
        input_df = pd.DataFrame([data])
        input_df = input_df[FEATURE_NAMES]  # ضمان الترتيب الصحيح للأعمدة
    except Exception as e:
        st.error(f"خطأ أثناء تجهيز البيانات: {e}")
        st.stop()

    # 4d. التحجيم القياسي (Standard Scaling - مثل الخلية 20)
    try:
        scaled_features = scaler.transform(input_df)
    except Exception as e:
        st.error(f"خطأ أثناء تطبيق الـ Scaler: {e}")
        st.write("البيانات قبل التحجيم القياسي (بعد التحجيم اليدوي):")
        st.dataframe(input_df)
        st.stop()

    # 4e. التنبؤ (Predict - مثل الخلية 27)
    try:
        prediction_normalized = model.predict(scaled_features)
    except Exception as e:
        st.error(f"خطأ أثناء التنبؤ: {e}")
        st.stop()

    # 4f. إلغاء التحجيم اليدوي للنتيجة (Un-normalize)
    # *** هذا هو السطر الذي تم تصحيحه ***
    # لأن النموذج يتنبأ بالقيمة المحجمة (التي قُسمت على 10 في الخلية 16)
    prediction_actual = prediction_normalized[0] * max_values['YearsAtCompany']

    # 4g. عرض النتيجة
    pred_rounded = int(round(prediction_actual))
    st.success(f"**Predicted years of service in the company:** `{pred_rounded:.2f}` years")

    if prediction_actual < 0:
        st.warning("التنبؤ أقل من 0. قد يشير هذا إلى أن المدخلات غير شائعة أو أن النموذج يحتاج لمراجعة.")

