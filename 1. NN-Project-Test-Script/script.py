import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import pandas as pd
import numpy as np


def load_model(model_path='densenet-model.h5'):
    print('========================')
    print('[WAIT]: Loading model...')
    model = keras.models.load_model(model_path)
    print('[INFO]: Model is Loaded!')
    print('========================\n')
    return model


def read_to_dataframe(test_dir='Test'):
    img_path = os.listdir(test_dir)
    test_df = pd.DataFrame({'image_name': img_path})
    print('========================')
    print("Number of Loaded Test Data Samples: ", test_df.shape[0])
    return test_df


def get_predictions(model=None, test_df=None, test_dir='Test'):
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    n_test_samples = test_df.shape[0]

    test_datagen = ImageDataGenerator(rescale=1/255.0)
    test_generator = test_datagen.flow_from_dataframe(test_df,
                                                      directory=test_dir,
                                                      x_col='image_name',
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      y_col=None,
                                                      batch_size=1,
                                                      class_mode=None,
                                                      shuffle=False)

    print('========================')
    pred_array = model.predict(test_generator, steps=n_test_samples)
    predictions = np.argmax(pred_array, axis=1)
    test_df['label'] = predictions
    print(test_df.head())
    print('========================')

    return test_df


def generate_prediction_file(test_df=None):
    print("[START]: File Creation...")
    test_df.to_csv(r'./[CS_22]-predictions.csv', index=False)
    print("[END]: File Creation!")


def run_script():
    print('=================================================')
    MODEL_PATH = 'densenet-model.h5'
    TEST_DIR = 'Test'

    # Loading DenseNet Model...
    densenet_model = load_model(MODEL_PATH)

    # Reading Test Images into a Dataframe...
    test_df = read_to_dataframe(TEST_DIR)

    # Image Data Generator for Testing...
    test_df = get_predictions(model=densenet_model, test_df=test_df, test_dir=TEST_DIR)

    # Submission File Creation...
    generate_prediction_file(test_df=test_df)
    print('=================================================')


if __name__ == '__main__':
    run_script()
