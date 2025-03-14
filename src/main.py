try:
    import streamlit as st
    import numpy as np
    import pickle
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as e:
    print("Required module not found. Install using: pip install streamlit numpy scikit-learn")
    raise e

# Streamlit Title
st.title("🏠 House Price Prediction App")

# User Inputs
sqft = st.number_input("Square Feet", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
age = st.number_input("House Age (Years)", min_value=0, max_value=100, value=10)

# Input Array
input_features = np.array([[sqft, bedrooms, bathrooms, age]])
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_features)

# Model Training
X_train = np.array([[1000, 2, 1, 5], [2000, 4, 3, 20], [1500, 3, 2, 10], [1200, 2, 2, 8], [1800, 3, 2, 15]])
y_train = np.array([150000, 300000, 220000, 180000, 250000])
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    st.write(f"### Predicted Price: ${prediction[0]:,.2f}")
