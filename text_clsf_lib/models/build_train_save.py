from text_clsf_lib.models.model_data import prepare_model_data
from text_clsf_lib.models.model_trainer_runner import NNModelTrainer, NNModelRunner


def build_train_save(preset: dict,
          save_tflite_model=False):
    """
    This function wraps up the whole training process.
    It includes the following steps configured by preset:
    - preparing model data (train/test).
    - training process.
    - saving model.
    - converting to tflite.

    :param preset: dict.
    :param save_tflite_model: bool
    :return: NNModelRunner instance.
    """
    data = prepare_model_data(
        data_params=preset['data_params'],
        vectorizer_params=preset['vectorizer_params'])

    model_trainer = get_model_trainer(preset)

    data_train = data['train_vectorized']

    model_trainer.train(data_train)
    print('Training process complete!')
    model_trainer.save(preset['model_save_dir'], preset['model_name'])
    if save_tflite_model:
        model_trainer.save_tflite(preset['model_save_dir'], preset['model_name'])
    print(f'Model saved to: {preset["model_save_dir"]}. Model name: {preset["model_name"]}')

    return NNModelRunner(model=model_trainer.model)


def get_model_trainer(preset: dict):
    return NNModelTrainer(
        model_builder_class=preset['model_builder_class'],
        architecture_params=preset['architecture_params'],
        vectorizer_params=preset['vectorizer_params'],
        training_params=preset['training_params']
    )
