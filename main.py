from pathlib import Path
from typing import *
import pandas as pd
import joblib

from flask import Flask
from flask_restx import Api, Resource
from werkzeug.datastructures import FileStorage

import models


app = Flask(__name__)
api = Api(app)

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

feature_columns = ['age', 'embarked', 'pclass', 'sex']
target_column = 'survived'
all_columns = feature_columns + [target_column]

upload_parser.add_argument('model_type',
                           required=True,
                           location='args',
                           choices=list(models.model_dict.keys()),
                           help='Bad choice: {error_msg}')

upload_parser.add_argument('model_params',
                           required=False,
                           location='application/json',
                           help='Bad choice: {error_msg}')

val_error_code = 400


@api.route('/train', methods=['PUT'],
           doc={'description': 'Train choosen model on part of the Titanic dataset, NaNs will be dropped'})
@api.expect(upload_parser)
class Train(Resource):
    @api.doc(params={
        'file': f'Excel file with columns: {*all_columns,}',
        'model_type': 'Model type',
        'model_params': 'Model hyperparameters in json format, check sklearn documentation of each model to know possible values',
    })
    @api.response(200, 'Success')
    @api.response(val_error_code, 'Validation Error')
    def put(self):
        args = upload_parser.parse_args()
        model_type = args['model_type']
        model_params = args['model_params']

        try:
            X, y = validate_and_prepare_data(args['file'], train=True)
        except ValueError as e:
            return val_error_code, str(e)

        model = models.create_model(model_type, model_params)
        model.fit(X, y)

        joblib.dump(model, model_dir / f'{model_type}.pkl')
        return f'Training successful on {X.shape[0]} samples'


@api.route('/predict', methods=['POST'],
           doc={'description': 'Predict the data in excel format, NaNs in data will raise Validation Error'})
@api.expect(upload_parser)
class Predict(Resource):
    @api.doc(params={
        'file': f'Excel file with columns: {*feature_columns,}',
        'model_type': 'Model type',
    })
    @api.response(200, 'Success')
    @api.response(val_error_code, 'Validation Error')
    def post(self):
        args = upload_parser.parse_args()
        model_type = args['model_type']
        model_fp = model_dir / f'{model_type}.pkl'
        if not model_fp.exists():
            return val_error_code, "Model hasn't been fitted"
        model = joblib.load(model_fp)

        try:
            X = validate_and_prepare_data(args['file'], train=False)
        except ValueError as e:
            return val_error_code, str(e)

        pred = model.predict(X)
        return {'prediction': pred.tolist()}


@api.route('/models_list', methods=['GET'],
           doc={'description': 'Get the list of saved trained models'})
class ModelsList(Resource):
    @staticmethod
    def get():
        model_names = [fp.name.replace('.pkl', '') for fp in model_dir.glob('*.pkl')]
        if not model_names:
            model_names = None
        else:
            model_names = ', '.join(model_names)

        return f'Trained models: {model_names}'


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


if __name__ == '__main__':
    app.run(debug=True)

