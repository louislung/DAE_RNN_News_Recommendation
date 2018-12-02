from sklearn.metrics import pairwise
from sklearn.preprocessing import normalize
import numpy as np

def pairwise_similarity(in_df, norm='', metric='cosine', set_diagonal_zero=True, **metric_param):
    """ Read articles data and map a positive and a negative article for every article

    :param in_df: each row is one item (e.g. article), each column is a feature (e.g. word)
    :param norm: norm option of sklearn.preprocessing import normalize, e.g. l1, l2, or max
    :param metric: similarity measure, e.g. cosine, dot product
    :param set_diagonal_zero: set diagonal to zero before return (as similarity of article itself always = 1, may not be useful)

    :type in_df: numpy.ndarray / scipy sparse metric / list
    :type norm: str
    :type metric: str
    :type set_diagonal_zero: boolean

    :return: pairwise similarity metric
    :rtype: if metric = 'dot product' and type of in_df != list:
                rtype = type of in_df
            else:
                numpy.ndarray

    .. note: linear kernal is simply dot product
             norm='l2' and metric='linear kernal' is actually equivalent to cosine similarity
    """
    assert metric in ['cosine','linear kernel']
    out_df = np.array([])
    metric_mapping = {
        'cosine': pairwise.cosine_similarity,
        'linear kernel': pairwise.linear_kernel,
        'euclidean': pairwise.euclidean_distances,
        'manhattan': pairwise.manhattan_distances
    }

    if norm != '':
        in_df = normalize(in_df, norm=norm)

    out_df = metric_mapping[metric](in_df,**metric_param)

    if set_diagonal_zero:
        np.fill_diagonal(out_df, 0)

    return out_df


if __name__ == '__main__':
    import scipy.sparse as sp
    list_cnt1 = [[1,1,0,1],[0,1,0,1],[0,1,1,1]]
    sparse_cnt1 = sp.csr_matrix(list_cnt1)
    ndarray_cnt1 = np.array(list_cnt1)
    result_cnt1 = np.array([[0., 0.816496580927726, 0.6666666666666669],[0.816496580927726, 0., 0.816496580927726],[0.6666666666666669, 0.816496580927726, 0.]])

    assert np.array_equal(pairwise_similarity(list_cnt1), result_cnt1)
    assert np.array_equal(pairwise_similarity(sparse_cnt1), result_cnt1)
    assert np.array_equal(pairwise_similarity(ndarray_cnt1), result_cnt1)

