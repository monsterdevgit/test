import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Loading the saved models
diabetes_model = pickle.load(open('./diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('./heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('./parkinsons_model.sav', 'rb'))


class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


class HeartDiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class ParkinsonsInput(BaseModel):
    fo: float
    fhi: float
    flo: float
    Jitter_percent: float
    Jitter_Abs: float
    RAP: float
    PPQ: float
    DDP: float
    Shimmer: float
    Shimmer_dB: float
    APQ3: float
    APQ5: float
    APQ: float
    DDA: float
    NHR: float
    HNR: float
    RPDE: float
    DFA: float
    spread1: float
    spread2: float
    D2: float
    PPE: float


@app.post("/predict_diabetes")
def predict_diabetes(data: DiabetesInput):
    diab_prediction = diabetes_model.predict([[data.Pregnancies, data.Glucose, data.BloodPressure,
                                               data.SkinThickness, data.Insulin, data.BMI,
                                               data.DiabetesPedigreeFunction, data.Age]])

    result = {"prediction": "The test shows signs of diabetes. We advice further medical attention is be required." if diab_prediction[0] == 1 else "Congratulations! The test results indicate that you are not diabetic."}
    return result

@app.post("/predict_heart_disease")
def predict_heart_disease(data: HeartDiseaseInput):
    heart_prediction = heart_disease_model.predict([[data.age, data.sex, data.cp, data.trestbps,
                                                    data.chol, data.fbs, data.restecg, data.thalach,
                                                    data.exang, data.oldpeak, data.slope, data.ca, data.thal]])

    result = {"prediction": "The test shows signs of heart disease in you. It is advisable to seek medical attention." if heart_prediction[0] == 1 else
              "Good news! The test shows no signs of heart disease"}
    return result

@app.post("/predict_parkinsons")
def predict_parkinsons(data: ParkinsonsInput):
    parkinsons_prediction = parkinsons_model.predict([[data.fo, data.fhi, data.flo, data.Jitter_percent,
                                                       data.Jitter_Abs, data.RAP, data.PPQ, data.DDP,
                                                       data.Shimmer, data.Shimmer_dB, data.APQ3, data.APQ5,
                                                       data.APQ, data.DDA, data.NHR, data.HNR, data.RPDE,
                                                       data.DFA, data.spread1, data.spread2, data.D2, data.PPE]])

    result = {"prediction": "The analysis indicates that you are showing potential signs of Parkinson's disease. It is highly recommended to consult a medical professional for a comprehensive evaluation." if parkinsons_prediction[0] == 1 else
              "The results indicate that you are not diagnosed with Parkinson's disease."}
    return result