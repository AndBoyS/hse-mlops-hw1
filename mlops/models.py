from typing import Optional, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline

model_dict = {
    'logistic_regression': LogisticRegression,
    'svm': LinearSVC,
}


def create_model(model_type: str,
                 model_params: Optional[Dict[str, Any]] = None
                 ) -> Pipeline:
    """
    Создать пайплайн модели

    :param model_type: тип модели, logistic_regression или svm
    :param model_params: словарь с гиперпараметрами
    :return: sklearn.pipeline.Pipeline
    """
    assert model_type in model_dict

    if model_params is None:
        model_params = {}
    model = model_dict[model_type](**model_params)

    return make_pipeline(
        ColumnTransformer([
            ('ohe', OneHotEncoder(), ['pclass', 'embarked']),
            ('binarizer', OrdinalEncoder(), ['sex'])
            ],
            remainder='passthrough'),
        model
    )
