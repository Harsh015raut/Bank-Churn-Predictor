import os 
import sys
import numpy as np 
import pandas as pd 
from dataclasses import dataclass 

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from src.exception import CustomException 
from src.logger import logging 
from sklearn.metrics import accuracy_score
from src.utils import save_object,evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split Training and test data.")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models ={
                        'LogisticRegression': LogisticRegression(),
                        'GaussianNB': GaussianNB(),
                        'KNeighborsClassifier': KNeighborsClassifier(),
                        'DecisionTreeClassifier': DecisionTreeClassifier(),
                        'XGBClassifier': XGBClassifier(),
                        'RandomForestClassifier': RandomForestClassifier(),
                        'GradientBoostingClassifier':GradientBoostingClassifier(),
                    }
            

            params ={
                        'LogisticRegression': {'C':[1]},
                        'GaussianNB': {'var_smoothing': np.logspace(0,-9, num=100)},
                        'KNeighborsClassifier': {'n_neighbors':[1,2,3,4,5,6,7],'weights':['uniform', 'distance'],'algorithm':['auto', 'ball_tree', 'kd_tree']},
                        'DecisionTreeClassifier': {'criterion': ['gini', 'entropy'],'max_depth': [5, 10, 15],'min_samples_split': [2, 5, 10]},
                        'XGBClassifier': {'learning_rate': [0.01, 0.1, 0.2],'max_depth': [3, 5, 7],'n_estimators': [50, 100, 200]},
                        'RandomForestClassifier': {'n_estimators':[10,25,50,100],'criterion':['gini', 'entropy', 'log_loss'],'max_depth':[1,3,5,7]},
                        'GradientBoostingClassifier': {'loss':['log_loss','exponential'],'learning_rate':[0.01,0.1,0.2],
                                                       'criterion':['friedman_mse', 'squared_error'],'max_depth':[1,3,5,7]}
                    }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test,y_test = y_test,
                                             models=models,param=params)
            
            ##To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing data")

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

            predicted = best_model.predict(X_test)

            accu_score = accuracy_score(y_test,predicted)
            return accu_score
    
        except Exception as e:
            raise CustomException(e,sys)
            
