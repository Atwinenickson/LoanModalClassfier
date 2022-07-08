# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


# cur_dir = os.path.dirname(__file__)

# clf = pickle.load(open(os.path.join(cur_dir, 'finall_modell.sav'), 'rb'))


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import os
from pydantic import BaseModel

import pickle as pi
import pandas as pd

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

# model_dir = "final_model.sav"
# model = load_model(model_dir)

cur_dir = os.path.dirname(__file__)

model = pi.load(open(os.path.join(cur_dir, 'final_model.sav'), 'rb'))


class Loan(BaseModel):
    Loan_ID: str
    Gender: str
    Married: bool
    Dependents: int
    Education: str
    Self_Employed: bool
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: str
    Credit_History: str
    Property_Area: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Loan Predictor API!"}


@app.post("/loanprediction/")
async def get_loan_prediction(item: Loan):
    Loan_ID = item.Loan_ID
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
    df = pd.DataFrame([[Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
                        Loan_Amount_Term, Credit_History, Property_Area]], columns=["Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                                                                                    "Loan_Amount_Term", "Credit_History", "Property_Area"], dtype=str, index=['input'])
    # vec = final_transformer.transform(df)
    my_prediction = model.predict(df)
    print(my_prediction)

    # return {
    #     "model-prediction": my_prediction
    # }

    return item
