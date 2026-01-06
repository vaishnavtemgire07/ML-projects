import os
import sys
from dataclasses import dataclass

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)       
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

# DecisionTreeRegressor was already imported above (duplicate import removed)

from source.exception import CustomException
from source.logger import logging 
from source.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Add optional models only if their packages are available
            if XGBRegressor is not None:
                models["XGBRegressor"] = XGBRegressor()
            else:
                logging.warning("XGBoost package not found; skipping XGBRegressor")

            if CatBoostRegressor is not None:
                models["CatBoosting Regressor"] = CatBoostRegressor(verbose=False)
            else:
                logging.warning("CatBoost package not found; skipping CatBoosting Regressor")
            
            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            # To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            
            # To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            logging.info(f"Best model found on both training and testing dataset: {best_model_name}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            logging.info("Exception occurred in the model trainer component")
            raise CustomException(e, sys)                                                               