# internal modules
import os
import datetime
import logging

# https://stackoverflow.com/a/38645250
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import tensorflow_datasets as tfds


BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
DISTRIBUTED = True


# initialise logging
logger = logging.getLogger("train_example")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# console logger
console_log = logging.StreamHandler()
console_log.setLevel(logging.WARNING)
console_log.setFormatter(formatter)
logger.addHandler(console_log)


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label


# Define a function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


def create_dataset(batch_size_per_replica: int, datasets, strategy):
    # process, shuffle, and batch training data
    batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_dataset, eval_dataset

def create_model(num_classes: int=10):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])


def train(
    n_epochs: int,
    num_classes: int,
    train_data,
    val_data,
    strategy
):
    # enter tensorflow distributed strategy
    if DISTRIBUTED:
        with strategy.scope():
            model = get_model(num_classes)
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy']
            )
    else:
        model = get_cnn()
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )

    # Define the checkpoint directory to store the checkpoints.
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]

    logger.info(f"Training model for {n_epochs} epochs")
    model.fit(train_dataset, epochs=n_epochs, callbacks=callbacks)


# Define a callback for printing the learning rate at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'\nLearning rate for epoch {epoch + 1} is {model.optimizer.lr.numpy()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Keras model."
    )
    parser.add_argument(
        "-n",
        "--n_epochs",
        type=int,
        default=12,
        help="Number of training epochs",
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
    # parse command line arguments
    args = parser.parse_args()
    c_log.setLevel(level=args.loglevel)  # logging console handler

    assert isinstance(args.n_epochs, int) and args.n_epochs > 0

    # check available GPU devices
    logger.debug("Available devices:")
    for i, device in enumerate(tf.config.list_logical_devices()):
        logger.debug(f"({i}) {device}")

    if DISTRIBUTED:
        # define distribution strategy for multiple workers
        strategy = tf.distribute.MirroredStrategy()
        batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
        logger.info(f'Number of devices: {strategy.num_replicas_in_sync}')
    else:
        strategy = None
        batch_size = BATCH_SIZE_PER_REPLICA
        logger.info("DISTRIBUTED=FALSE")

    # download dataset
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    num_classes = info.features["label"].num_classes

    train_data, val_data = get_dataset(batch_size_per_replica, datasets, strategy)

    strategy = tf.distribute.MirroredStrategy()

    train(args.n_epochs, num_classes, train_data, val_data, strategy)