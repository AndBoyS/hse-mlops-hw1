from flask import Flask
from flask_restx import Api, Resource
from werkzeug.datastructures import FileStorage

from mlops import ml_models, data, app_model
from mlops.error_protocol import val_error_code, success_code


app = Flask(__name__)
api = Api(app)

train_parser = api.parser()
train_parser.add_argument('file', location='files',
                          type=FileStorage, required=True)
train_parser.add_argument('model_type',
                          required=True,
                          location='args',
                          choices=list(ml_models.model_dict.keys()),
                          help='Bad choice: {error_msg}')
train_parser.add_argument('model_params',
                          required=False,
                          location='application/json',
                          help='Bad choice: {error_msg}')

predict_parser = api.parser()
predict_parser.add_argument('file', location='files',
                            type=FileStorage, required=True)
predict_parser.add_argument('model_name',
                            required=True,
                            location='args')


@api.route('/train', methods=['PUT'],
           doc={'description': 'Train choosen model on part of the Titanic dataset, NaNs will be dropped'})
@api.expect(train_parser)
class Train(Resource):
    @api.doc(params={
        'file': f'Excel file with columns: {*data.all_columns,}',
        'model_type': 'Model type',
        'model_params': 'Model hyperparameters in json format, check sklearn documentation of each model to know possible values',
    })
    @api.response(success_code, 'Success')
    @api.response(val_error_code, 'Validation Error')
    def put(self):
        args = train_parser.parse_args()

        train_data, train_target = app_model.get_data(args['file'], train=True)

        return app_model.fit(train_data, train_target,
                             args['model_type'],
                             args['model_params'])


@api.route('/predict', methods=['POST'],
           doc={'description': 'Predict the data in excel format, NaNs in data will raise Validation Error'})
@api.expect(predict_parser)
class Predict(Resource):
    @api.doc(params={
        'file': f'Excel file with columns: {*data.feature_columns,}',
        'model_name': 'Name of the model to predict with',
    })
    @api.response(success_code, 'Success')
    @api.response(val_error_code, 'Validation Error')
    def post(self):
        args = predict_parser.parse_args()
        train_data = app_model.get_data(args['file'], train=False)
        pred = app_model.predict(train_data, args['model_name'])
        return {'prediction': pred.tolist()}


@api.route('/models_list', methods=['GET'],
           doc={'description': 'Get the list of saved trained models'})
class ModelsList(Resource):
    @staticmethod
    def get():
        model_names = app_model.get_available_model_names()
        return {'trained_models': model_names}
