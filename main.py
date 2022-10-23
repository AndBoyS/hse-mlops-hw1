import pandas as pd
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

feature_columns = ['age', 'embarked', 'pclass', 'sex']
target_column = 'survived'

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
class MlMethods(Resource):
    @api.doc(params={'file': 'Excel file'})
    def put(self):
        args = upload_parser.parse_args()
        data = pd.read_excel(args['file'])
        X, y = self.prepare_data(data, train=True)
        model.fit(X, y)
        return 'Training successful'

    @staticmethod
    def prepare_data(df, train=True):
        if train:
            df = df[feature_columns + [target_column]].dropna()
        else:
            assert not df.isna().sum(), 'Nan data found'

        X, y = df[feature_columns], df[target_column]
        return X, y


if __name__ == '__main__':
    app.run(debug=True)

