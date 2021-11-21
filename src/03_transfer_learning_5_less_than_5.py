import argparse
import os
<<<<<<< HEAD
#import shutil
#from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
#import random
import tensorflow as tf
import numpy as np
import io

STAGE = "transfer learning even odd" ## <<< change stage name 



=======
import numpy as np
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import tensorflow as tf
import io

STAGE = "transfer learning" ## <<< change stage name 
>>>>>>> 11d9f49677d93c7e10fcadaa476c3196604cdeed

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def update_greater_than_less_than_5(list_of_labels):
    for idx, label in enumerate(list_of_labels):
        even_condition = label%2 == 0 ## CHANGE THE CONDITION
        list_of_labels[idx] = np.where(even_condition, 1, 0)
    return list_of_labels


<<<<<<< HEAD
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

    ## change to odd and even
    y_train_bin, y_test_bin, y_valid_bin = update_greater_than_less_than_5([y_train, y_test, y_valid])

    ## set the seeds
    seed = 2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## log the model summary in logs
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    ## load the base model
    base_model_path = os.path.join("artifacts", "models", "base_model.h5")
    base_model = tf.keras.models.load_model(base_model_path)

    logging.info(f"loaded base model summary: \n{_log_model_summary(base_model)}")
    logging.info(f"base model evaluation metrics: \n{base_model.evaluate(X_test, y_test_bin)}") # << y_test_bin for our usecase


=======
def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    
    ## get the data
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


    y_train_bin, y_test_bin, y_valid_bin = update_even_odd_labels([y_train, y_test, y_valid])

    ## set the seeds
    seed = 2021 ## get it from config
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## log our model summary information in logs
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    ## load the base model - 
    base_model_path = os.path.join("artifacts", "models", "base_model.h5")
    base_model = tf.keras.models.load_model(base_model_path)
    logging.info(f"loaded base model summary: \n{_log_model_summary(base_model)}")

    ## freeze the weights
>>>>>>> 11d9f49677d93c7e10fcadaa476c3196604cdeed
    for layer in base_model.layers[: -1]:
        print(f"trainable status of before {layer.name}:{layer.trainable}")
        layer.trainable = False
        print(f"trainable status of after {layer.name}:{layer.trainable}")

    base_layer = base_model.layers[: -1]
    # ## define the model and compile it
    new_model = tf.keras.models.Sequential(base_layer)
    new_model.add(
        tf.keras.layers.Dense(2, activation="softmax", name="output_layer")
    )

<<<<<<< HEAD

    logging.info(f"{STAGE}: model summary: \n{_log_model_summary(new_model)}")       

    # No need to define layers, use base_model
    # LAYERS = [
    #         tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
    #         tf.keras.layers.Dense(300, name="hiddenLayer1"),
    #         tf.keras.layers.LeakyReLU(),
    #         tf.keras.layers.Dense(100, name="hiddenLayer2"),
    #         tf.keras.layers.LeakyReLU(),
    #         tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")]
    # ## define the model and compile its
    # model = tf.keras.models.Sequential(LAYERS)

    # Compile the model
    LOSS_FUNCTION = "sparse_categorical_crossentropy" # for classification model
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]

    new_model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    # ## log the model summary in logs
    # def _log_model_summary(model):
    #     with io.StringIO() as stream:
    #         model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
    #         summary_str = stream.getvalue()
    #     return summary_str

#    model.summary()
    logging.info(f"transfer model summary: \n{_log_model_summary(new_model)}")
=======
    logging.info(f"{STAGE} model summary: \n{_log_model_summary(new_model)}")

    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]

    new_model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS) 
>>>>>>> 11d9f49677d93c7e10fcadaa476c3196604cdeed


    ## Train the model
    history = new_model.fit(
<<<<<<< HEAD
        X_train, y_train_bin,
        epochs=10,
        validation_data=(X_valid, y_valid_bin), # << y_valid_bin for our usecase
        verbose=2)

    ## save the base model 
    model_dir_path = os.path.join("artifacts","models")
    create_directories([model_dir_path])
    
    model_file_path = os.path.join(model_dir_path, "even_odd_model.h5")
    new_model.save(model_file_path)

=======
        X_train, y_train_bin, # << y_train_bin for our usecase
        epochs=10, 
        validation_data=(X_valid, y_valid_bin), # << y_valid_bin for our usecase
        verbose=2)

    ## save the base model - 
    model_dir_path = os.path.join("artifacts","models")
    model_file_path = os.path.join(model_dir_path, "even_odd_model.h5")
    new_model.save(model_file_path)

    logging.info(f"base model is saved at {model_file_path}")
    logging.info(f"evaluation metrics {new_model.evaluate(X_test, y_test_bin)}")  # << y_test_bin for our usecase
>>>>>>> 11d9f49677d93c7e10fcadaa476c3196604cdeed

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
<<<<<<< HEAD
#    args.add_argument("--params", "-p", default="params.yaml")
=======
>>>>>>> 11d9f49677d93c7e10fcadaa476c3196604cdeed
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e