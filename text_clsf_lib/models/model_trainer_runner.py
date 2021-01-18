import os
import numpy as np
from keras import Model
from keras.models import load_model, save_model
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from tensorflow.lite.python.lite import TFLiteConverter
from io import StringIO
import sys
import shutil
TARGET_NAMES = ['Offenseless', 'Offensive']


class NNModelTrainer:
    """
    Class used for model training.
    Takes in model builder class and preset configuration.

    """
    def  __init__(self, model_builder_class, architecture_params, vectorizer_params, training_params):
        model_builder = model_builder_class(architecture_params, vectorizer_params)
        self.training_params = training_params
        self.model: Model = model_builder.prepare_model_architecture()

    def train(self, train_data) -> None:
        X_train, y_train = train_data
        y_train = to_categorical(y_train)
        self.model.fit(x=X_train,
                       y=y_train,
                       batch_size=self.training_params['batch_size'],
                       epochs=self.training_params['epochs'],
                       validation_split=self.training_params['validation_split'],
                       callbacks=self.training_params['callbacks'])

    def test(self, test_data, show_sklearn_report=True) -> tuple:
        return test_model(self.model, test_data, show_sklearn_report)

    def save(self, save_dir, model_name):
        os.makedirs(save_dir, exist_ok=True)
        save_dir = f'{save_dir}'
        model_path = f'{save_dir}/{model_name}.h5'
        save_model(self.model, model_path)
        self.save_model_summary(save_dir, model_name)

    def save_tflite(self, save_dir, model_name):
        self.model.save(f'{save_dir}', save_format='tf')
        converter = TFLiteConverter.from_saved_model(f'{save_dir}')
        tflite_model = converter.convert()
        with open(f'{save_dir}/{model_name}.tflite', 'wb') as f:
            f.write(tflite_model)

    def save_model_summary(self, save_dir, model_name):
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        self.model.summary()
        with open(f"{save_dir}/{model_name}_summary.txt", 'w') as f:
            mystdout.seek(0)
            shutil.copyfileobj(mystdout, f)
        sys.stdout = old_stdout


class NNModelRunner:
    """
    This class is used after training process.
    Can be loaded either from a filepath or by using existing model.
    """
    def __init__(self, model=None, model_path=None):
        self.model: Model = model if model is not None else load_model(model_path)

    def test(self, test_data, show_sklearn_report=False):
        return test_model(self.model, test_data, show_sklearn_report)

    def run(self, data: list or np.array):
        return self.model.predict(data)


def test_model(model, test_data, show_sklearn_report=False) -> tuple:
    X_test, y_test = test_data
    predictions = model.predict(X_test)
    pred_labels = [np.argmax(pred) for pred in predictions]
    if show_sklearn_report:
        print(classification_report(y_test, pred_labels, target_names=TARGET_NAMES))
    return predictions, y_test
