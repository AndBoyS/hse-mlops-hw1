from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

model_dict = {
    'logistic_regression': LogisticRegression,
    'svm': LinearSVC,
}


def create_model(model_type: str):
    """
    Создать пайплайн модели

    :param model_type: тип модели, logistic_regression или svm
    :return:
    """
    assert model_type in model_dict

    return make_pipeline(
        ColumnTransformer([
            ('ohe', OneHotEncoder(), ['pclass', 'embarked']),
            ('binarizer', OrdinalEncoder(), ['sex'])
            ],
            remainder='passthrough'),
        model_dict[model_type]()
    )