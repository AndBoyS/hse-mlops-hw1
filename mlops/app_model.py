from pathlib import Path
from typing import Optional, List, Union, Tuple

import joblib
import numpy as np
import pandas as pd

from werkzeug.datastructures import FileStorage

from mlops import data, ml_models
from mlops.error_protocol import val_error_code


model_dir = Path('../models')
model_dir.mkdir(exist_ok=True)


def fit(train_data: pd.DataFrame,
        train_target: pd.Series,
        model_type: str,
        model_params: dict) -> str:
    model = ml_models.create_model(model_type, model_params)
    model.fit(train_data, train_target)

    joblib.dump(model, model_dir / f'{model_type}.pkl')
    return f'Training successful on {train_data.shape[0]} samples'


def predict(train_data: pd.DataFrame, model_type: str) -> np.array:

    model_fp = model_dir / f'{model_type}.pkl'
    if not model_fp.exists():
        return val_error_code, "Model hasn't been fitted"
    model = joblib.load(model_fp)

    pred = model.predict(train_data)
    return pred


def get_data(file: FileStorage, train: bool = True
             ) -> Union[Tuple[int, str], Tuple[pd.DataFrame, pd.Series]]:
    """
    Подготовливает данные (data.validate_and_prepare_data) и при ошибке возвращает val_error_code и ошибку
    :param file:
    :param train:
    :return:
    """
    try:
        train_data, train_target = data.validate_and_prepare_data(file, train=train)
    except ValueError as e:
        return val_error_code, str(e)
    return train_data, train_target


def get_available_model_names() -> Optional[List[str]]:
    """
    Получить список моделей, которые обучены
    :return:
    """
    model_names = [fp.name.replace('.pkl', '') for fp in model_dir.glob('*.pkl')]
    if not model_names:
        model_names = None
    else:
        model_names = ', '.join(model_names)

    return model_names
