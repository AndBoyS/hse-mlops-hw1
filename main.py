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


@api.route('/train', methods=['PUT'])
@api.expect(upload_parser)
class Train(Resource):
    @api.doc(params={
        'file': f'Excel file with columns: {*all_columns,}',
        'model_type': 'Model type',
        'model_params': 'Model hyperparameters in json format, check sklearn documentation of each model to know possible values',
    })
    def put(self):
        args = upload_parser.parse_args()
        model_type = args['model_type']
        model_params = args['model_params']
        data = pd.read_excel(args['file'])
        X, y = self.prepare_data(data)
        model = models.create_model(model_type, model_params)
        model.fit(X, y)

        joblib.dump(model, model_dir / f'{model_type}.pkl')
        return f'Training successful on {X.shape[0]} samples'

    @staticmethod
    def prepare_data(df: pd.DataFrame,
                     ) -> Tuple[pd.DataFrame, pd.Series]:
        df = df[all_columns].dropna()
        return df[feature_columns], df[target_column]


@api.route('/predict', methods=['POST'])
@api.expect(upload_parser)
class Predict(Resource):
    @api.doc(params={
        'file': f'Excel file with columns: {*feature_columns,}',
        'model_type': 'Model type',
    })
    def post(self):
        args = upload_parser.parse_args()
        model_type = args['model_type']
        model_fp = model_dir / f'{model_type}.pkl'
        if not model_fp.exists():
            return 400, "Model hasn't been fitted"
        model = joblib.load(model_fp)

        data = pd.read_excel(args['file'])
        X = data[feature_columns]

        if X.isna().sum().sum():
            return 400, 'Nans found in data'

        pred = model.predict(X)
        return {'prediction': pred.tolist()}


@api.route('/models_list', methods=['GET'])
class ModelsList(Resource):
    @staticmethod
    def get():
        model_names = [fp.name.replace('.pkl', '') for fp in model_dir.glob('*.pkl')]
        if not model_names:
            model_names = None
        else:
            model_names = ', '.join(model_names)

        return f'Trained models: {model_names}'


if __name__ == '__main__':
    app.run(debug=True)

