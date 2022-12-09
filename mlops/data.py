from typing import Union, Tuple

import pandas as pd


feature_columns = ['age', 'embarked', 'pclass', 'sex']
target_column = 'survived'
all_columns = feature_columns + [target_column]


def validate_and_prepare_data(file, train: bool = True
                              ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """
    Проверяет файл с данными и возвращает их в формате pandas

    :param file:
    :param train:
    :return: X, y если train, X если not train
    """
    try:
        data = pd.read_excel(file)
    except Exception as e:
        raise ValueError(f"Couldn't read file: {e}")

    needed_columns = feature_columns
    if train:
        needed_columns = all_columns

    if not set(needed_columns).issubset(data.columns):
        raise ValueError('Not all columns in data')

    data = data[needed_columns]

    if train:
        data.dropna(inplace=True)
        return data[feature_columns], data[target_column]

    X = data[feature_columns]
    if X.isna().sum().sum():
        raise ValueError('Data contains NaNs')

    return X
