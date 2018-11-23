from scipy import sparse
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def pairwise_similarity(in_df, metric='cosine'):
    """ Read articles data and map a positive and a negative article for every article

    :param in_df: each row is one item (e.g. article), each column is a feature (e.g. word)
    :param metric: similarity measure, e.g. cosine

    :type in_df: pandas (Sparse)DataFrame / numpy.ndarray / scipy sparse metric / list
    :type metric: str

    :return: pairwise similarity with diagonal set to 0
    :rtype: numpy.ndarray
    """
    if sparse.issparse(in_df):
        out_df = cosine_similarity(in_df)
        np.fill_diagonal(out_df, 0)
        return out_df
    else:
        return squareform(1-pdist(in_df, metric))
