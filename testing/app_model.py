import unittest
from pathlib import Path

from mlops import app_model


repo_dir = Path().resolve().parent
data_dir = repo_dir / 'example_data'


class MyTestCase(unittest.TestCase):

    model_type = 'logistic_regression'
    model_params = {'C': 1}
    model_params_str = app_model.dict_to_str(model_params)
    model_name = f'{model_type}_{model_params_str}'
    model_fp = Path('models') / f'{model_name}.pkl'

    def test_dict_to_str(self):
        d = {'key1': 1, 'key2': '2'}
        d_str = app_model.dict_to_str(d)
        d_str_expected = 'key1_1_key2_2'
        self.assertEqual(d_str, d_str_expected)

    def test_fit(self):
        train_data_fp = data_dir / 'titanic_train.xlsx'

        train_data, train_target = app_model.get_data(train_data_fp)

        app_model.fit(train_data, train_target, self.model_type, self.model_params)

        self.assertTrue(self.model_fp.exists())

    def test_predict(self):
        test_data_fp = data_dir / 'titanic_test.xlsx'

        test_data = app_model.get_data(test_data_fp, train=False)

        pred = app_model.predict(test_data, self.model_name)
        self.assertEqual(pred.shape[0], test_data.shape[0])


if __name__ == '__main__':
    unittest.main()
