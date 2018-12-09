from sklearn.metrics import pairwise
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
import numpy as np, pandas as pd
from matplotlib import pyplot as plt, font_manager as mfm


font_path = "/System/Library/Fonts/PingFang.ttc"
fontproperties = mfm.FontProperties(fname=font_path)


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


def visualize_scatter(data_2d, label, title, figsize=(20, 20), save_path=None):
    plt.figure(figsize=figsize)
    plt.grid()

    label_factorizer = pd.factorize(label)
    label_ids = label_factorizer[0]

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=plt.cm.gist_ncar((label_id + 1) / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=label_factorizer[1][label_id])
    plt.legend(loc='best', prop=fontproperties)

    if title is not None:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)


if __name__ == '__main__':
    import scipy.sparse as sp
    list_cnt1 = [[1,1,0,1],[0,1,0,1],[0,1,1,1]]
    sparse_cnt1 = sp.csr_matrix(list_cnt1)
    ndarray_cnt1 = np.array(list_cnt1)
    result_cnt1 = np.array([[0., 0.816496580927726, 0.6666666666666669],[0.816496580927726, 0., 0.816496580927726],[0.6666666666666669, 0.816496580927726, 0.]])

    assert np.array_equal(pairwise_similarity(list_cnt1), result_cnt1)
    assert np.array_equal(pairwise_similarity(sparse_cnt1), result_cnt1)
    assert np.array_equal(pairwise_similarity(ndarray_cnt1), result_cnt1)

