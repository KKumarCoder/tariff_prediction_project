from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the trained model and other components
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    country = request.form["country"].strip().title()
    product_category = request.form["product_category"].strip().title()
    tariff_percentage = float(request.form["tariff_percentage"])
    export_loss = float(request.form["export_loss"])
    gdp_impact = float(request.form["gdp_impact"])
    employment_loss = int(request.form["employment_loss"])

    # Encode categorical values
    if country in label_encoders["country"].classes_:
        country_encoded = label_encoders["country"].transform([country])[0]
    else:
        return render_template("index.html", error=f"❌ Country '{country}' not found in dataset.")

    if product_category in label_encoders["product_category"].classes_:
        product_encoded = label_encoders["product_category"].transform([product_category])[0]
    else:
        return render_template("index.html", error=f"❌ Product category '{product_category}' not found in dataset.")

    # Create input data array
    user_data = np.array([[country_encoded, product_encoded, tariff_percentage, export_loss, gdp_impact, employment_loss]])

    # Scale input data
    user_data_scaled = scaler.transform(user_data)

    # Predict impact level
    prediction = model.predict(user_data_scaled)
    predicted_label = label_encoders["impact_level"].inverse_transform(prediction)[0]

    return render_template("result.html", impact_level=predicted_label)

if __name__ == "__main__":
    app.run(debug=True)

