from pathlib import Path
from typing import *
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from flask import Flask
from flask_restx import Api, Resource
from werkzeug.datastructures import FileStorage


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

model = make_pipeline(
    ColumnTransformer([
        ('ohe', OneHotEncoder(), ['pclass', 'embarked']),
        ('binarizer', OrdinalEncoder(), ['sex'])
        ],
        remainder='passthrough'),
    LinearRegression()
)


@api.route('/train', methods=['PUT'])
@api.expect(upload_parser)
class Train(Resource):
    @api.doc(params={'file': f'Excel file with columns: {*all_columns,}'})
    def put(self):
        args = upload_parser.parse_args()
        data = pd.read_excel(args['file'])
        X, y = self.prepare_data(data)
        model.fit(X, y)
        joblib.dump(model, model_dir / 'model.pkl')
        return 'Training successful'

    @staticmethod
    def prepare_data(df: pd.DataFrame,
                     ) -> Tuple[pd.DataFrame, pd.Series]:
        df = df[all_columns].dropna()
        return df[feature_columns], df[target_column]


@api.route('/predict', methods=['POST'])
@api.expect(upload_parser)
class Predict(Resource):
    @api.doc(params={'file': f'Excel file with columns: {*feature_columns,}'})
    def post(self):

        model_fp = model_dir / 'model.pkl'
        if not model_fp.exists():
            return 400, "Model hasn't been fitted"
        model = joblib.load(model_fp)

        args = upload_parser.parse_args()
        data = pd.read_excel(args['file'])
        X = data[feature_columns]

        if X.isna().sum().sum():
            return 400, 'Nans found in data'

        pred = model.predict(X)
        return {'prediction': pred.tolist()}


if __name__ == '__main__':
    app.run(debug=True)

