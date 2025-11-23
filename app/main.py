from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Load the trained model
model = joblib.load("app/model.joblib")

# 2. Define Input Data Structure (Data Validation)
# This ensures the API rejects bad data automatically.
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 3. Initialize API
app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML Model is Live"}

@app.post("/predict")
def predict(data: IrisRequest):
    # Convert input data to numpy array
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    # Make prediction
    prediction = model.predict(features)

    # Return result
    class_name = ["setosa", "versicolor", "virginica"][int(prediction[0])]
    return {"prediction": class_name}