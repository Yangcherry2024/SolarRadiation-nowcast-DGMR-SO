import tensorflow as tf


class Loss_hing_disc():
    def __init__(self) -> None:
        pass

    def __call__(self, score_generated, score_real):
        """Discriminator hinge loss."""
        l1 = tf.nn.relu(1. - score_real)
        loss = tf.reduce_mean(l1)  # , axis=list(range(1, len(l1.shape)))
        l2 = tf.nn.relu(1. + score_generated)
        loss += tf.reduce_mean(l2)  # , axis=list(range(1, len(l2.shape)))
        tf.print("Debugging: Disc Loss: ", loss)
        return loss


class Loss_hing_gen():
    def __init__(self) -> None:
        pass

    def __call__(self, score_generated):
        """Generator hinge loss."""
        loss = - \
            tf.reduce_mean(
                score_generated)  # ,axis=list(range(1, len(score_generated.shape)))
        tf.print("Debugging, Gen Loss: ", loss)
        return loss
