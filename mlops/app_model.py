import re
from pathlib import Path
from typing import Optional, List, Union, Tuple
import json
import joblib
import numpy as np
import pandas as pd

from werkzeug.datastructures import FileStorage

from mlops import data, ml_models
from mlops.error_protocol import val_error_code


model_dir = Path('models')
model_dir.mkdir(exist_ok=True)


def dict_to_str(d: dict) -> str:
    """
    Преобразует словарь в строковый вид, который можно вставлять в название файла
    {'a': 1, 'b': 2} -> 'a_1_b_2'
    """
    s = json.dumps(d)
    s = s.replace(':', '_')
    s = s.replace(',', '_')
    s = re.sub(r'\W', '', s)
    return s


def fit(train_data: pd.DataFrame,
        train_target: pd.Series,
        model_type: str,
        model_params: Optional[dict]) -> str:
    '''
    Обучает модель и сохраняет ее в model_dir/{model_type}_{params_str}.pkl
    '''

    if not model_params:
        model_params = ml_models.default_params_dict[model_type]

    model = ml_models.create_model(model_type, model_params)
    model.fit(train_data, train_target)

    params_str = dict_to_str(model_params)
    print(params_str, model_params)
    model_fp = model_dir / f'{model_type}_{params_str}.pkl'

    joblib.dump(model, model_fp)
    return f'Training successful on {train_data.shape[0]} samples'


def predict(test_data: pd.DataFrame, model_name: str) -> np.array:

    model_fp = model_dir / f'{model_name}.pkl'
    if not model_fp.exists():
        return val_error_code, "Model hasn't been fitted"
    model = joblib.load(model_fp)

    pred = model.predict(test_data)
    return pred


def get_data(
        file: Union[str, Path, FileStorage],
        train: bool = True,
        ) -> Union[Tuple[int, str], Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """
    Подготовливает данные (data.validate_and_prepare_data) и при ошибке возвращает val_error_code и ошибку
    :param file:
    :param train:
    :return:
    """
    try:
        res = data.validate_and_prepare_data(file, train=train)
    except ValueError as e:
        return val_error_code, str(e)
    return res


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


def delete_model(model_name: str):
    model_fp = model_dir / f'{model_name}.pkl'
    model_fp.unlink()
