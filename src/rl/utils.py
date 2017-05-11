import numpy as np
import tensorflow as tf


def one_hot(a, n, multiple_agent=True):
    x = np.zeros(shape=a.shape+(n,))
    if multiple_agent:
        x[(range(len(a)), a)] = 1
    else:
        x[a] = 1
    return x


def huber_loss(x, max_grad=1.):
    raw_loss = tf.abs(x)
    return tf.where(
      tf.less(raw_loss, max_grad),
      tf.multiply(tf.square(raw_loss), 0.5),
      tf.subtract(tf.multiply(raw_loss, max_grad), 0.5*max_grad*max_grad)
    )


def mean_huber_loss(x, max_grad=1.):
    return tf.reduce_mean(huber_loss(x, max_grad))


def mean_squared_loss(x):
    return tf.reduce_mean(tf.square(x))


def process_samples(samples, n, action_discrete=True, multiple_agent=True):
    s = np.array(list(map(lambda s: s.state, samples)))
    ns = np.array(list(map(lambda s: s.next_state, samples)))
    r = np.array(list(map(lambda s: s.reward, samples)))
    # If the actions aren't discrete, they are stored in replay buffer as vector
    if action_discrete:
        a = np.array(list(map(lambda s: one_hot(s.action, n, multiple_agent), samples)))
    else:
        a = np.array(list(map(lambda s: s.action, samples)))
    t = np.array(list(map(lambda s: s.is_terminal, samples)))
    return (s, a, ns, r, t)
