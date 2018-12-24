from ..utils import xavier_init
from ..utils import gen_batches
from ..utils import gen_batches_triplet
from ..utils import masking_noise
from ..utils import salt_and_pepper_noise
from ..utils import decay_noise
from ..utils import get_sparse_ind_val_shape
import numpy as np, pandas as pd, tensorflow as tf
from scipy import sparse

def test_gen_batches():
    num_data = 30
    data = np.array([x for x in range(num_data)]).reshape((-1,1)).astype(np.float32)
    data_corrupted = np.random.randint(0,2,(num_data, 10)).astype(np.float32)
    data_label = np.random.randint(0,10, num_data).astype(np.float32)

    # Test when label is
    # 1. None
    # 2. 1-d np array
    # 3. 2-d np array
    # 4. pd Series
    # 5. pd DataFrame
    for label in [None, data_label, data_label.reshape((-1,1)), pd.Series(data_label), pd.DataFrame(data_label)]:
        # Test when data & data_corrupted is
        # 1. 2-d np array
        # 2. csr_matrix
        # 3. pd DataFrame
        for func in [lambda x: x, sparse.csr_matrix, pd.DataFrame]:
            in_data = func(data)
            in_data_corrupted = func(data_corrupted)

            if isinstance(in_data, pd.DataFrame):
                in_data.index = np.random.choice(num_data*2, num_data,replace=False)
                in_data_corrupted.index = in_data.index

            if isinstance(label, pd.DataFrame) or isinstance(label, pd.Series):
                label.index = np.random.choice(num_data*2, num_data,replace=False)

            for batch_size in [4,0.3]:
                row_show = np.zeros(num_data)
                for results in gen_batches(in_data, in_data_corrupted, batch_size=batch_size, data_label=label):
                    a = results[0]
                    b = results[1]
                    if sparse.issparse(a):
                        a = a.toarray()
                        b = b.toarray()

                    if isinstance(in_data, pd.DataFrame):
                        idx = a.loc[:,0].astype(int).tolist()
                        assert (data_corrupted[idx, :] == b).all(axis=None, skipna=False)
                    else:
                        idx = list(a[:,0].astype(int))
                        assert (data_corrupted[idx, :] == b).all()

                    if label is not None:
                        if isinstance(label, pd.DataFrame) or isinstance(label, pd.Series):
                            assert (label.iloc[idx] == results[2]).all(axis=None, skipna=False)
                        else:
                            assert (label[idx] == results[2]).all()
                    row_show[idx] += 1
                assert row_show.sum() == num_data

def test_gen_batches_triplet():
    num_data = 30
    data = {'org': np.array([x for x in range(num_data)]).reshape((-1, 1)).astype(np.float32),
            'pos': np.array([x for x in range(num_data)]).reshape((-1, 1)).astype(np.float32),
            'neg': np.array([x for x in range(num_data)]).reshape((-1, 1)).astype(np.float32),
            }
    data_corrupted = {'org': np.random.randint(0, 2, (num_data, 10)).astype(np.float32),
                      'pos': np.random.randint(0, 2, (num_data, 10)).astype(np.float32),
                      'neg': np.random.randint(0, 2, (num_data, 10)).astype(np.float32),
                      }
    in_data = data.copy()
    in_data_corrupted = data_corrupted.copy()

    # Test when data & data_corrupted is
    # 1. 2-d np array
    # 2. csr_matrix
    # 3. pd DataFrame todo
    for func in [lambda x: x, sparse.csr_matrix]:
        row_show = np.zeros(num_data)
        for key in data:
            in_data[key] = func(data[key])
            in_data_corrupted[key] = func(data_corrupted[key])
        # if isinstance(in_data, pd.DataFrame):
        #     for key in data:
        #         in_data[key].index = np.random.choice(num_data, num_data, replace=False)
        #         in_data_corrupted[key].index = in_data.index

        for results in gen_batches_triplet(in_data, in_data_corrupted, batch_size=4):
            a = results[0]
            b = results[1]
            if sparse.issparse(a[0]):
                a = [i.toarray() for i in results[0]]
                b = [i.toarray() for i in results[1]]

            assert (a[0] == a[1]).all()
            assert (a[0] == a[2]).all()
            idx = list(a[0][:, 0].astype(int))
            i = 0
            for key in data_corrupted:
                assert (data_corrupted[key][idx, :] == b[i]).all()
                i += 1

            row_show[idx] += 1
        assert row_show.sum() == num_data

def test_masking_noise():
    num_data = 10
    X = sparse.csr_matrix(np.random.rand(num_data, 10000).astype(np.float32))

    for in_X in [X, X.toarray()]:
        for prob in [0., 0.3, 1.]:
            X_masked = masking_noise(in_X, prob)
            X_masked = sparse.csr_matrix(X_masked)

            if prob == 0.:
                assert (X != X_masked).nnz == 0
            elif prob == 1.:
                assert X_masked.nnz == 0
            else:
                assert X_masked.nnz / X.nnz - (1. - prob) <= 1e-2
                # assert data is corrupted, no new data is created
                for i in range(num_data-1):
                    assert set(X_masked.indices[X_masked.indptr[0]:X_masked.indptr[i+1]]) <= set(X.indices[X.indptr[0]:X.indptr[i+1]])

def test_salt_and_pepper_noise():
    assert 1==1

def test_decay_noise():
    assert 1==1

def test_get_sparse_ind_val_shape():
    num_data = 10
    X = sparse.coo_matrix(np.random.randint(0, 5, (num_data, 3)).astype(np.float32))

    sparse_tf = tf.sparse.placeholder('float')
    X_tf = tf.Session().run(tf.sparse.to_dense(sparse_tf), {sparse_tf: get_sparse_ind_val_shape(X)})

    assert (X.toarray() == X_tf).all()

