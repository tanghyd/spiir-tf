import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# https://stackoverflow.com/a/38645250
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import tensorflow as tf
import tensorflow_datasets as tfds

# initialise logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# console logger
console_log = logging.StreamHandler()
console_log.setLevel(logging.WARNING)
console_log.setFormatter(formatter)
logger.addHandler(console_log)


def convert_loglevel(loglevel: int) -> int:
    """Keras fit verbose:

    verbose=0 will show you nothing (silent)
    verbose=1 will show you an animated progress bar
    verbose=2 will just print the number of epochs every line:

    """
    if args.loglevel <= logging.DEBUG:
        verbose = 2
    elif args.loglevel <= logging.INFO:
        verbose = 1
    else:
        verbose = 0
    return verbose


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


# Define a function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch: int) -> float:
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


def create_dataset(
    batch_size_per_replica: int,
    datasets: Dict[str, tf.data.Dataset],
    strategy: Optional[tf.distribute.Strategy],
    buffer: int = 10000,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    # determine batch size
    num_replicas = strategy.num_replicas_in_sync if strategy is not None else 1
    batch_size = batch_size_per_replica * num_replicas

    # process, shuffle, and batch training data
    train_data = datasets["train"].map(scale).cache().shuffle(buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data = datasets["test"].map(scale).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_data, val_data


def create_model(num_classes: int = 10):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )


def train(
    n_epochs: int,
    num_classes: int,
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    strategy: Optional[tf.distribute.Strategy],
    verbose: Union[int, bool],
):
    # enter tensorflow distributed strategy
    if strategy is not None:
        with strategy.scope():
            model = create_model(num_classes)
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"],
            )
    else:
        model = create_model(num_classes)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

    # define early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        verbose=verbose,
    )
    # define the checkpoint directory to store the checkpoints.
    checkpoint_dir = "./training_checkpoints"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}"),
        save_weights_only=True,
        verbose=verbose,
    )

    # combine callbacks
    callbacks = [
        checkpoint_callback,
        early_stopping_callback,
        tf.keras.callbacks.LearningRateScheduler(decay),
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    history = model.fit(
        train_data,
        epochs=n_epochs,
        callbacks=callbacks,
        validation_data=val_data,
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Keras model.")
    parser.add_argument(
        "-n",
        "--n-epochs",
        type=int,
        default=25,
        metavar="N",
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size-per-replica",
        type=int,
        default=64,
        metavar="N",
        help="input batch size per GPU for training (default: 64)",
    )
    parser.add_argument(
        "--distribute",
        action="store_true",
        dest="distribute",
        help="Run distributed TensorFlow (default: False)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Specify data directory to download data set.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
        help="Display all developer debug logging statements",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
        help="Set logging level to INFO and display progress and information",
    )
    args = parser.parse_args()
    console_log.setLevel(level=args.loglevel)  # logging console handler
    start_time = time.perf_counter()

    # check input types
    assert isinstance(args.n_epochs, int) and args.n_epochs > 0
    assert isinstance(args.batch_size_per_replica, int) and args.batch_size_per_replica > 0

    # check available GPU devices
    logger.debug("Available devices:")
    for i, device in enumerate(tf.config.list_logical_devices()):
        logger.debug(f"({i}) {device}")

    # define distribution strategy for multiple workers
    strategy = tf.distribute.MirroredStrategy() if args.distribute else None

    # download dataset
    datasets, info = tfds.load(name="mnist", with_info=True, as_supervised=True, data_dir=args.data_dir)
    num_classes = info.features["label"].num_classes
    train_data, val_data = create_dataset(args.batch_size_per_replica, datasets, strategy)

    # train model
    train(
        n_epochs=args.n_epochs,
        num_classes=num_classes,
        train_data=train_data,
        val_data=val_data,
        strategy=strategy,
        verbose=convert_loglevel(args.loglevel),
    )

    duration = round(time.perf_counter() - start_time, 4)
    logger.info(f"train.py completed in {duration}s.")
