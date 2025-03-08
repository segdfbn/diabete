from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI application
app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

# Global variables
training_loss = []

# Data URL and columns
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Load dataset
def load_data():
    return pd.read_csv(DATA_URL, names=COLUMNS)

# Create necessary directories
os.makedirs("static/images", exist_ok=True)

# Data model for patient input
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Callback to track training loss
class LossHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        training_loss.append(logs['loss'])
        plt.figure()
        plt.plot(training_loss, label='Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"static/images/loss_plot_epoch_{epoch + 1}.png")
        plt.close()

# Train the model
@app.post("/train/")
def train_model():
    global training_loss
    df = load_data()
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, fr"C:\Users\Admin\OneDrive\Bureau\dossier_seghira\diabete\.models\scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=2, validation_data=(X_test, y_test), callbacks=[LossHistory()])
    model.save(fr"C:\Users\Admin\OneDrive\Bureau\dossier_seghira\diabete\.models\diabetes_model.h5")

    return {"message": "Model training complete and saved."}

# Predict diabetes
@app.post("/predict/")
def predict(data: PatientData):
    if not (os.path.exists(fr"C:\Users\Admin\OneDrive\Bureau\dossier_seghira\diabete\.models\diabetes_model.h5") and os.path.exists(fr"C:\Users\Admin\OneDrive\Bureau\dossier_seghira\diabete\.models\scaler.pkl")):
        raise HTTPException(status_code=400, detail="Model or scaler not found. Please train the model first.")

    model = tf.keras.models.load_model(fr"C:\Users\Admin\OneDrive\Bureau\dossier_seghira\diabete\.models\diabetes_model.h5")
    scaler = joblib.load(fr"C:\Users\Admin\OneDrive\Bureau\dossier_seghira\diabete\.models\scaler.pkl")

    input_data = np.array([[
        data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
        data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "Diabetic" if prediction[0][0] >= 0.5 else "Not Diabetic"

    return {"prediction": result, "confidence": float(prediction[0][0])}

# Home page
@app.get("/api")
def home():
    return {"message": "Welcome to the Diabetes Detection API!"}

# Web form for user input
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Predict from web form
@app.post("/predict-web/")
async def predict_web(
    Pregnancies: int = Form(...),
    Glucose: float = Form(...),
    BloodPressure: float = Form(...),
    SkinThickness: float = Form(...),
    Insulin: float = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...)
):
    data = PatientData(
        Pregnancies=Pregnancies, Glucose=Glucose, BloodPressure=BloodPressure,
        SkinThickness=SkinThickness, Insulin=Insulin, BMI=BMI,
        DiabetesPedigreeFunction=DiabetesPedigreeFunction, Age=Age
    )
    prediction = predict(data)
    return {"Prediction": prediction["prediction"], "Confidence": prediction["confidence"]}

# Get training loss data
@app.get("/get-training-loss/")
def get_training_loss():
    return {"training_loss": training_loss}

# Display loss plot
@app.get("/get-loss-plot/{epoch}")
def get_loss_plot(epoch: int):
    image_path = f"static/images/loss_plot_epoch_{epoch}.png"
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail="Loss plot not found.")
