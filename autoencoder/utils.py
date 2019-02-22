from scipy import sparse
import tensorflow as tf
import numpy as np, os, pandas as pd
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
    

def gen_batches(data, data_corrupted, batch_size, data_label=None, random=True):
    """ Divide input data into batches.

    :param data: input scipy sparse matrix / numpy array / pd DataFrame
    :param data_corrupted: input corrupted data, same type as data
    :param batch_size: size of each batch, (0,1] or integer >=1
    :param data_label: label of data, pd DataFrame / pd Series / numpy 1-d, or 2-d array

    :return: data divided into batches

    ..  note: data & data_corrupted must be of same type, but data_label can be any data type
    """
    assert batch_size > 0.
    assert data.shape[0] == data_corrupted.shape[0]
    assert type(data) == type(data_corrupted),(type(data), type(data_corrupted))
    if isinstance(data, pd.DataFrame): assert (data.index == data_corrupted.index).all()
    if data_label is not None: assert data_label.ndim == 1 or data_label.shape[1] == 1

    if batch_size < 1.: batch_size = max(round(data.shape[0] * batch_size), 1)
    batch_size = int(batch_size)

    index = list(range(0, data.shape[0]))
    if random: np.random.shuffle(index)

    for i in range(0, data.shape[0], batch_size):
        if isinstance(data, pd.DataFrame):
            batch_data = data.iloc[index[i:i + batch_size]]
            batch_data_corrupted = data_corrupted.iloc[index[i:i + batch_size]]

        else:
            batch_data = data[index[i:i + batch_size]]
            batch_data_corrupted = data_corrupted[index[i:i + batch_size]]

        if data_label is not None:
            if isinstance(data_label, pd.DataFrame) or isinstance(data_label, pd.Series):
                batch_label = data_label.iloc[index[i:i + batch_size]]
            else:
                batch_label = data_label[index[i:i + batch_size]]
            yield (batch_data, batch_data_corrupted, batch_label)

        else:
            yield (batch_data, batch_data_corrupted)


def gen_batches_triplet(data, data_corrupted, batch_size, random=True):
    """ Divide input data into batches.

    :param data: dictionary of input data which can be sparse matrix / np 2-d array BUT NOT pd DataFrame
    :param data_corrupted: dictionary of input corrupted data
    :param batch_size: size of each batch

    :return: data divided into batches
    """
    assert batch_size > 0.
    for key in data:
        assert data[key].shape[0] == data_corrupted[key].shape[0]

    if batch_size < 1.: batch_size = max(round(data[key].shape[0] * batch_size), 1)
    index = list(range(0, data[key].shape[0]))
    if random: np.random.shuffle(index)

    for i in range(0, data[key].shape[0], batch_size):
        yield [data[key][index[i:i+batch_size],:] for key in data], [data_corrupted[key][index[i:i+batch_size],:] for key in data]


def masking_noise(X, v):
    """ Apply masking noise to data in X, in other words a fraction v of elements of X
    (chosen at random) is forced to zero.

    :param X: np 2-d array or sparse matrix
    :param v: float, corruption rate

    :return: transformed data
    """
    assert 0. <= v <= 1.

    X_noise = X.tocoo(True) if not isinstance(X, np.ndarray) else X.copy()

    if isinstance(X, np.ndarray):
        mask = np.random.choice(a=[0, 1], size=X_noise.shape, p=[v, 1 - v])
        X_noise = mask * X_noise
    else:
        mask = np.random.rand(X_noise.nnz) >= v
        X_noise.row = X_noise.row[mask]
        X_noise.col = X_noise.col[mask]
        X_noise.data = X_noise.data[mask]
    return X_noise.tocsr() if not isinstance(X, np.ndarray) else X_noise


def salt_and_pepper_noise(X, v):
    """ Apply salt and pepper noise to data in X, in other words a fraction v of elements of X
    (chosen at random) is set to its maximum or minimum value according to a fair coin flip.
    If minimum or maximum are not given, the min (max) value in X is taken.

    :param X: array_like, Input data
    :param v: int, fraction of elements to distort

    :return: transformed data
    """
    X_noise = X.tolil(True) if not isinstance(X, np.ndarray) else X.copy()
    n_features = X.shape[1]

    mn = X.min()
    mx = X.max()

    for i, sample in enumerate(X):
        mask = np.random.randint(0, n_features, v)

        for m in mask:

            if np.random.random() < 0.5:
                X_noise[i,m] = mn
            else:
                X_noise[i,m] = mx

    return X_noise.tocsr() if not isinstance(X, np.ndarray) else X_noise


def decay_noise(X, v):
    """ Apply decaying noise to data in X, in other words all elements of X is decayed by a fraction v

    :param X: array_like, Input data
    :param v: float, corruption rate

    :return: transformed data
    """
    X_noise = X.copy()

    X_noise = X_noise * (1. - v)

    return X_noise


def get_sparse_ind_val_shape(sparse_m):
    """ get indices, values, shape of a sparse matrix for feeding tf sparse placeholder

    :param sparse_m: input sparse matrix

    :type any scipy sparse matrix, csr/csc/coo/lil

    :return: tuple of indices, values, shape
    """
    if not isinstance(sparse_m, sparse.csr_matrix):
        sparse_m = sparse.csr_matrix(sparse_m)
    sparse_m.sort_indices()

    sparse_m = sparse.coo_matrix(sparse_m)
    indices = np.column_stack((sparse_m.row, sparse_m.col))
    values = sparse_m.data
    shape = sparse_m.shape

    return (indices, values, shape)

