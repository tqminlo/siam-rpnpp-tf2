import tensorflow as tf


def loss_func(y_true, y_pred):
    cls_true, cls_pred = y_true[0], y_pred[0]
    loc_true, loc_pred = y_true[1], y_pred[1]

    # cls loss
    cls_loss = cls_true * tf.math.log(cls_pred + 0.0001) + (1. - cls_true) * tf.math.log(1.0001 - cls_pred)
    cls_loss = - tf.reduce_mean(cls_loss)

    # loc_loss
    loc_loss = tf.abs(loc_true - loc_pred)
    loc_loss = tf.reduce_sum(loc_loss, axis=-1)
    loc_loss = tf.reduce_sum(loc_loss)

    loss = cls_loss + loc_loss
    return loss
