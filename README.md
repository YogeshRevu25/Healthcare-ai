# 🏥 Healthcare Cost & Claims Analysis (AI + ML)

This project analyzes medical insurance data to understand **key factors influencing healthcare costs** (like age, BMI, and smoking habits) and uses a **Machine Learning model** to predict estimated medical charges.  
It is deployed as a **Streamlit Web App** so users can interactively explore data and get personalized cost predictions.

---

## 📌 Project Overview
- **Dataset:** `insurance.csv` (1338 records, 7 columns)
- **Goal:** Identify factors affecting healthcare costs and predict medical charges.
- **Tech Stack:** Python, Pandas, Seaborn, Scikit-learn, Matplotlib, Streamlit
- **ML Model:** Random Forest Regressor (trained on preprocessed features)
- **Deployment:** Streamlit Community Cloud

---

## 📊 Features & Visualizations
✅ **Exploratory Data Analysis (EDA)**  
- Distribution of charges by **smoker vs non-smoker**
- Relationship between **BMI & charges**
- Regional differences in healthcare costs
- Correlation heatmap of all features

✅ **Machine Learning Prediction**  
- Users can input:
  - Age
  - BMI
  - Number of Children
  - Smoker/Non-smoker
  - Region
  - Sex  
- The app predicts **estimated medical charges** instantly.

✅ **Interactive Dashboard**  
- Clean Streamlit interface  
- Each visualization has title + description  
- CSV cleaned and downloadable  

---

## 🖼 Screenshots

### 1️⃣ App Homepage  
![App Homepage](assets/homepage.png)

### 2️⃣ Visualization Example  
![Visualization](assets/visualization.png)

### 3️⃣ Prediction Form  
![Prediction](assets/prediction.png)

---

## 🚀 Live Demo
🔗 **Try the App Here:** [Click to Open](https://yogeshrevu25-healthcare-ai.streamlit.app)  
_(Link will work after deployment)_

---

## ⚙️ Installation & Run Locally

```bash
# Clone this repository
git clone https://github.com/Yogeshrevu25/healthcare-ai.git
cd healthcare-ai

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
