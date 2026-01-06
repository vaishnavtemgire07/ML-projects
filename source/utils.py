import os
import sys
import dill

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from source.exception import CustomException

def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models: dict) -> dict:
    try:
        report = {}

        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Calculate R2 score using sklearn for correctness
            r2_square = float(r2_score(y_test, y_test_pred))

            report[model_name] = round(r2_square, 4)

        return report

    except Exception as e:
        raise CustomException(e, sys)
