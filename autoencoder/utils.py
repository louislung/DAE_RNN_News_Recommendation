from scipy import misc
import tensorflow as tf
import numpy as np, os
from pathlib import Path


# Define path
_script_path = Path(os.path.dirname(os.path.realpath(__file__)))


# ############# #
#   Utilities   #
# ############# #


def xavier_init(fan_in, fan_out, const=1):
    """ Xavier initialization of network weights.
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

    :param fan_in: fan in of the network (n_features)
    :param fan_out: fan out of the network (n_components)
    :param const: multiplicative constant
    """
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)
    

def gen_batches(data, data_corrupted, batch_size, sparse):
    """ Divide input data into batches.

    :param data: input data
    :param data_corrupted: input corrupted data
    :param batch_size: size of each batch
    :param sparse: whether input data is sparse matrix e.g. csr_matrix

    :return: data divided into batches
    """
    assert data.shape[0] == data_corrupted.shape[0]
    index = list(range(0, data.shape[0]))
    np.random.shuffle(index)

    for i in range(0, data.shape[0], batch_size):
        yield (data[index[i:i+batch_size],:], data_corrupted[index[i:i+batch_size],:])


def gen_batches_triplet(data, data_corrupted, batch_size, sparse):
    """ Divide input data into batches.

    :param data: dictionary of input data
    :param data_corrupted: dictionary of input corrupted data
    :param batch_size: size of each batch
    :param sparse: whether input data is sparse matrix e.g. csr_matrix

    :return: data divided into batches
    """
    for key in data:
        assert data[key].shape[0] == data_corrupted[key].shape[0]
    index = list(range(0, data[key].shape[0]))
    np.random.shuffle(index)

    for i in range(0, data[key].shape[0], batch_size):
        yield [data[key][index[i:i+batch_size],:] for key in data], [data_corrupted[key][index[i:i+batch_size],:] for key in data]


def masking_noise(X, v):
    """ Apply masking noise to data in X, in other words a fraction v of elements of X
    (chosen at random) is forced to zero.

    :param X: array_like, Input data
    :param v: int, fraction of elements to distort

    :return: transformed data
    """
    X_noise = X.copy()

    n_samples = X.shape[0]
    n_features = X.shape[1]

    for i in range(n_samples):
        mask = np.random.randint(0, n_features, v)

        for m in mask:
            X_noise[i][m] = 0.

    return X_noise


def salt_and_pepper_noise(X, v):
    """ Apply salt and pepper noise to data in X, in other words a fraction v of elements of X
    (chosen at random) is set to its maximum or minimum value according to a fair coin flip.
    If minimum or maximum are not given, the min (max) value in X is taken.

    :param X: array_like, Input data
    :param v: int, fraction of elements to distort

    :return: transformed data
    """
    X_noise = X.copy()
    n_features = X.shape[1]

    mn = X.min()
    mx = X.max()

    for i, sample in enumerate(X):
        mask = np.random.randint(0, n_features, v)

        for m in mask:

            if np.random.random() < 0.5:
                X_noise[i][m] = mn
            else:
                X_noise[i][m] = mx

    return X_noise


def decay_noise(X, v):
    """ Apply decaying noise to data in X, in other words all elements of X is decayed by a fraction v

    :param X: array_like, Input data
    :param v: int, fraction of elements to distort

    :return: transformed data
    """
    X_noise = X.copy()

    X_noise = X_noise * v

    return X_noise


def gen_image(img, width, height, outfile, img_type='grey'):
    assert len(img) == width * height or len(img) == width * height * 3

    if img_type == 'grey':
        misc.imsave(outfile, img.reshape(width, height))

    elif img_type == 'color':
        misc.imsave(outfile, img.reshape(3, width, height))

def get_sparse_ind_val_shape(sparse_m):
    """ get indices, values, shape of a sparse matrix for feeding tf sparse placeholder

    :param sparse_m: input sparse matrix

    :return: tuple of indices, values, shape
    """

    indices = tuple(np.array(_) for _ in zip(*sorted(zip(*sparse_m.nonzero()))))
    values = np.array(sparse_m[indices]).reshape(-1)
    indices = np.matrix(indices).transpose()
    shape = sparse_m.shape

    return (indices, values, shape)

if __name__ == "__main__":
    #todo: some automatic testing here
    print('pending')