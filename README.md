# 🏠 House Price Prediction App

A simple **Streamlit-based House Price Prediction App**, fully Dockerized and ready to deploy.

DEMO - https://app-house-price-2jkjgudeh34m37mvnt435t.streamlit.app/

---

## 📂 Project Structure

```
streamlit-docker/
│── .streamlit/
│   └── config.toml
│── src/
│   └── main.py
│── Dockerfile
│── requirements.txt
│── README.md
```

---

## 🚀 Prerequisites

- Python 3.9+
- Docker
- VS Code
- GitHub Account

---

## 🎯 Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/streamlit-house-price.git
cd streamlit-house-price
```

---

## 📂 Step 2: Create .streamlit Config File

```bash
mkdir .streamlit
New-Item -ItemType File .streamlit\config.toml
```

### 📝 **.streamlit/config.toml**
```toml
[server]
headless = true
runOnSave = true
fileWatcherType = "poll"
```

---

## 📝 Step 3: Create Source Code File

```bash
mkdir src
New-Item -ItemType File src\main.py
```

### 📜 **src/main.py**
```python
try:
    import streamlit as st
    import numpy as np
    import pickle
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as e:
    print("Required module not found. Install using: pip install streamlit numpy scikit-learn")
    raise e

st.title("🏠 House Price Prediction App")

sqft = st.number_input("Square Feet", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
age = st.number_input("House Age (Years)", min_value=0, max_value=100, value=10)

input_features = np.array([[sqft, bedrooms, bathrooms, age]])
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_features)

X_train = np.array([[1000, 2, 1, 5], [2000, 4, 3, 20], [1500, 3, 2, 10], [1200, 2, 2, 8], [1800, 3, 2, 15]])
y_train = np.array([150000, 300000, 220000, 180000, 250000])
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    st.write(f"### Predicted Price: ${prediction[0]:,.2f}")
```

---

## 📦 Step 4: Add Dependencies

```bash
New-Item -ItemType File requirements.txt
```

### 📜 **requirements.txt**
```txt
streamlit
numpy
scikit-learn
```

---

## 🐳 Step 5: Create Dockerfile

```bash
New-Item -ItemType File Dockerfile
```

### 🐳 **Dockerfile**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
EXPOSE 8501
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🛠️ Step 6: Build Docker Image

```bash
docker build -t streamlit-app .
```

---

## 🌐 Step 7: Run Docker Container

```bash
docker run -p 8501:8501 streamlit-app
```

Open in Browser: [http://localhost:8501](http://localhost:8501)

---

## 🌲 Step 8: Push to GitHub

```bash
git init
git add .
git commit -m "Dockerized Streamlit House Price Prediction App"
git branch -M main
git remote add origin https://github.com/<your-username>/streamlit-house-price.git
git push -u origin main
```

---

## 🎉 Final Output

✅ Fully Dockerized Streamlit App  
✅ Runs on Localhost  
✅ Live GitHub Repo  

---

## 🛑 **Facing Any Error?**

Contact me on LinkedIn or GitHub. 😎

---

