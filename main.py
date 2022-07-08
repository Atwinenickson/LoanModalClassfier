from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import os
from pydantic import BaseModel

import pickle as pi
import pandas as pd
import numpy as np 
from sklearn.utils.validation import check_array
from sklearn.preprocessing import LabelEncoder
import json

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

cur_dir = os.path.dirname(__file__)

model = pi.load(open(os.path.join(cur_dir, 'final_model.sav'), 'rb'))
le = LabelEncoder()


class Loan(BaseModel):
    Loan_ID: object
    Gender: object
    Married: object
    Dependents: object
    Education: object
    Self_Employed: object
    ApplicantIncome: int
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: object


@app.get("/")
async def root():
    return {"message": "Welcome to the Loan Predictor API!"}


@app.post("/loanprediction/")
async def get_loan_prediction(item: Loan):
    Loan_ID = int(item.Loan_ID)
    Gender = item.Gender
    Married = item.Married
    Dependents = item.Dependents
    Education = item.Education
    Self_Employed = item.Self_Employed
    ApplicantIncome = item.ApplicantIncome
    CoapplicantIncome = item.CoapplicantIncome
    LoanAmount = item.LoanAmount
    Loan_Amount_Term = item.Loan_Amount_Term
    Credit_History = item.Credit_History
    Property_Area = item.Property_Area
    
    
    info = [Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
                        Loan_Amount_Term, Credit_History, Property_Area]
                        
    loan_data = le.fit_transform(info)
    print(loan_data)
    my_prediction = model.predict([loan_data])
    print(my_prediction)
    print(my_prediction.tolist())
    print(np.array(my_prediction))
    data = my_prediction.tolist()

    return {
        "model-prediction": data,
    }

    
