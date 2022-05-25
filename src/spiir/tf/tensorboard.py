import io
import matplotlib.pyplot as plt
import tensorflow as tf


def figure_to_image(figure: plt.Figure, fmt: str="png") -> tf.Tensor:
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.
    The supplied figure is closed and inaccessible after this call.

    https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data

    Parameters
    ----------
        figure
            A matplotlib.pyplot.Figure object to convert to a valid image for TensorBoard
    
    
    Returns
    -------
    tf.Tensor
        The image as a compatible TensorFlow tensor object.


    """
    # save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format=fmt)  # save figure to buffer in memory
    plt.close()  # prevent display on notebook
    buf.seek(0)

    # convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)  # add the batch dimension
    return image