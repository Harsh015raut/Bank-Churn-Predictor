import sys 
import pandas as pd 
from src.exception import CustomException 
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
        pass 

    def predict(self,features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 CreditScore:int,
                 Geography:str,
                 Gender:str,
                 Age:int,
                 Tenure:int,
                 Balance:int,
                 HasCrCard:int,
                 IsActiveMember:int,
                 EstimatedSalary:int
                 ):
        
        self.CreditScore = CreditScore
        self.Geography = Geography
        self.Gender = Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary

    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                "CreditScore": [self.CreditScore],
                "Geography": [self.Geography],
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Tenure": [self.Tenure],
                "Balance": [self.Balance],
                "HasCrCard": [self.HasCrCard],
                "IsActiveMember": [self.IsActiveMember],
                "EstimatedSalary": [self.EstimatedSalary],
                "NumOfProducts": [0]  # Add 'NumOfProducts' column with a default value
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
