import numpy as np
import tensorflow as tf


def one_hot(a, n):
    x = np.zeros(n)
    x[a] = 1
    return x


def huber_loss(x, max_grad=1.):
    raw_loss = tf.abs(x)
    return tf.maximum(
      tf.multiply(tf.square(raw_loss), 0.5),
      tf.subtract(tf.multiply(raw_loss, max_grad), 0.5*max_grad*max_grad)
    )


def mean_huber_loss(x, max_grad=1.):
    return tf.reduce_mean(huber_loss(x, max_grad))


def process_samples(samples, n):
    s = np.array(list(map(lambda s: s.state, samples)))
    ns = np.array(list(map(lambda s: s.next_state, samples)))
    r = np.array(list(map(lambda s: s.reward, samples)))
    a = np.array(list(map(lambda s: one_hot(s.action, n), samples)))
    t = np.array(list(map(lambda s: s.is_terminal, samples)))
    return (s, a, ns, r, t)

