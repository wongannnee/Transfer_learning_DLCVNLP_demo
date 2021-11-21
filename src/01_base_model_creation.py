import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import numpy as np
import io

STAGE = "STAGE_NAME" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

#def main(config_path, params_path):
def main(config_path):
    ## read config files
    config = read_yaml(config_path)
#    params = read_yaml(params_path)
    
    ## get the data from MNIST training
    # Prepare the validation data from the full training data
    # normalising divide 255, Train = 55,000, Validation =5,000, Test=10,000
    mnist = tf.keras.datasets.mnist
    (X_train_full,y_train_full),(X_test, y_test) = mnist.load_data()
    X_valid, X_train = X_train_full[:5000] / 255. , X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000] , y_train_full[5000:] 
    X_test = X_test / 255.

    ## set the seeds
    seed = 2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## define layers
    LAYERS = [
            tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
            tf.keras.layers.Dense(300, name="hiddenLayer1"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(100, name="hiddenLayer2"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")]
    
    ## defind the model and compile its
    model = tf.keras.models.Sequential(LAYERS)

    # Compile the model
    LOSS_FUNCTION = "sparse_categorical_crossentropy" # for classification model
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]

    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    ## log the model summary in logs
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

#    model.summary()
    logging.info(f"base model summary: \n{_log_model_summary(model)}")


    ## Train the model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_valid, y_valid),
        verbose=2)

    ## save the base model 
    model_dir_path = os.path.join("artifacts","models")
    create_directories([model_dir_path])
    
    model_file_path = os.path.join(model_dir_path, "base_model.h5")
    model.save(model_file_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
#    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e