import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Healthcare AI Dashboard", layout="wide")

# ------------------- Load Data -------------------
@st.cache_data
def load_data():
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "insurance.csv")
    df = pd.read_csv(csv_path)
    return df

df = load_data()

st.title("🏥 Healthcare AI Dashboard")
st.markdown("Explore healthcare data, visualize trends, and predict medical charges using Machine Learning.")

# ------------------- Data Overview -------------------
st.header("📊 Dataset Overview")
st.dataframe(df.head())

st.write(f"**Shape of Dataset:** {df.shape[0]} rows × {df.shape[1]} columns")
st.write("### Column Information")
st.write(df.describe())

# ------------------- Visualizations -------------------
st.header("📈 Data Visualizations")

# 1. Age Distribution
st.subheader("Age Distribution")
st.caption("Shows the spread of patients' ages.")
fig, ax = plt.subplots()
sns.histplot(df['age'], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# 2. Charges Distribution
st.subheader("Charges Distribution")
st.caption("Visualizing how medical charges are spread across patients.")
fig, ax = plt.subplots()
sns.histplot(df['charges'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# 3. Charges by Smoker Status
st.subheader("Charges by Smoker")
st.caption("Smokers have significantly higher medical charges compared to non-smokers.")
fig, ax = plt.subplots()
sns.boxplot(x='smoker', y='charges', data=df, ax=ax)
st.pyplot(fig)

# 4. Charges by Region
st.subheader("Charges by Region")
st.caption("Regional variation in charges.")
fig, ax = plt.subplots()
sns.barplot(x='region', y='charges', data=df, estimator=np.mean, ci=None, ax=ax)
st.pyplot(fig)

# ------------------- Feature Engineering -------------------
categorical_cols = ['sex', 'smoker', 'region']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ------------------- Train Model -------------------
X = df.drop(columns=['charges'])
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.header("🤖 Machine Learning Model")
st.metric("Mean Absolute Error (Lower is Better)", round(mae, 2))

# ------------------- Feature Importance -------------------
st.subheader("Feature Importance")
importances = model.feature_importances_
fig, ax = plt.subplots()
sns.barplot(x=importances, y=X.columns, ax=ax)
ax.set_xlabel("Importance")
st.pyplot(fig)

# ------------------- Prediction Form -------------------

# Encode input
# ---------------------------
# 🔮 PREDICTION SECTION
# ---------------------------

# ---------------------------
# 🔮 PREDICTION SECTION
# ---------------------------

st.header("💡 Predict Your Medical Charges")
st.write("Enter your details to get an estimated medical cost prediction:")

# Take user inputs
age = st.number_input("Age", min_value=0, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker?", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Prepare input data as DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# ✅ Apply the same encoding as training
input_data['sex'] = input_data['sex'].map({'male': 1, 'female': 0})
input_data['smoker'] = input_data['smoker'].map({'yes': 1, 'no': 0})
input_data['region'] = input_data['region'].map({
    'southeast': 0,
    'southwest': 1,
    'northeast': 2,
    'northwest': 3
})

# Ensure the same column order used during training
training_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
input_data = input_data[training_columns]

# Make prediction
if st.button("Predict Charges"):
    try:
        predicted_charge = model.predict(input_data)[0]
        st.success(f"💰 Estimated Medical Charges: **${predicted_charge:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
