from flask import Flask
from flask_restx import Api, Resource
from werkzeug.datastructures import FileStorage

from mlops import ml_models, data, app_model
from mlops.error_protocol import val_error_code

app = Flask(__name__)
api = Api(app)

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)
upload_parser.add_argument('model_type',
                           required=True,
                           location='args',
                           choices=list(ml_models.model_dict.keys()),
                           help='Bad choice: {error_msg}')
upload_parser.add_argument('model_params',
                           required=False,
                           location='application/json',
                           help='Bad choice: {error_msg}')


@api.route('/train', methods=['PUT'],
           doc={'description': 'Train choosen model on part of the Titanic dataset, NaNs will be dropped'})
@api.expect(upload_parser)
class Train(Resource):
    @api.doc(params={
        'file': f'Excel file with columns: {*data.all_columns,}',
        'model_type': 'Model type',
        'model_params': 'Model hyperparameters in json format, check sklearn documentation of each model to know possible values',
    })
    @api.response(200, 'Success')
    @api.response(val_error_code, 'Validation Error')
    def put(self):
        args = upload_parser.parse_args()

        X, y = app_model.get_data(args['file'], train=True)

        return app_model.fit(X, y,
                             args['model_type'],
                             args['model_params'])


@api.route('/predict', methods=['POST'],
           doc={'description': 'Predict the data in excel format, NaNs in data will raise Validation Error'})
@api.expect(upload_parser)
class Predict(Resource):
    @api.doc(params={
        'file': f'Excel file with columns: {*data.feature_columns,}',
        'model_type': 'Model type',
    })
    @api.response(200, 'Success')
    @api.response(val_error_code, 'Validation Error')
    def post(self):
        args = upload_parser.parse_args()
        X = app_model.get_data(args['file'], train=False)
        pred = app_model.predict(X, args['model_type'])
        return {'prediction': pred.tolist()}


@api.route('/models_list', methods=['GET'],
           doc={'description': 'Get the list of saved trained models'})
class ModelsList(Resource):
    @staticmethod
    def get():
        model_names = app_model.get_available_model_names()
        return f'Trained models: {model_names}'
