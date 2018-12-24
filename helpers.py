from sklearn.metrics import pairwise, auc, roc_curve
from sklearn.preprocessing import normalize
import numpy as np, pandas as pd, scipy.sparse as sparse
from matplotlib import pyplot as plt, font_manager as mfm, os


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


def visualize_pairwise_similarity(labels, pairwise_similarity_metrics, plot='boxplot', title=None, figsize=(16, 9), save_path=None, **plot_kwargs):
    """plot similarity of related and not related article based on labels

    :param labels: numeric array or vector, -1 represents missing value and will be filtered automatically
    :param pairwise_similarity_metrics:
    :return:
    """
    assert labels.shape[0] == pairwise_similarity_metrics.shape[0]
    assert pairwise_similarity_metrics.shape[0] == pairwise_similarity_metrics.shape[1]
    assert plot in ['scatter', 'boxplot']
    if labels.shape.__len__() == 1: labels = np.expand_dims(labels, 1)

    not_nan_mask = np.squeeze(np.logical_and(np.expand_dims(labels, 0) >= 0, np.expand_dims(labels, 1) >= 0))
    mask = np.squeeze(np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1)))
    mask = np.logical_and(mask, not_nan_mask)
    related_mask = sparse.coo_matrix(np.tril(mask, -1))
    related_data = pairwise_similarity_metrics[related_mask.row, related_mask.col]
    unrelated_mask = sparse.coo_matrix(np.tril(np.logical_and(np.logical_not(mask), not_nan_mask), -1))
    unrelated_data = pairwise_similarity_metrics[unrelated_mask.row, unrelated_mask.col]

    max_data_limit = int(1e7)
    if len(related_data) > max_data_limit:
        related_data = np.random.choice(related_data, max_data_limit,replace=False)
    if len(unrelated_data) > max_data_limit:
        unrelated_data = np.random.choice(unrelated_data, max_data_limit, replace=False)

    # Boxplot
    plt.figure(figsize=figsize)
    plt.subplot(121)
    if plot == 'scatter':
        plt.scatter(['Related']*len(related_data), related_data, **plot_kwargs)
        plt.scatter(['Unrelated']*len(unrelated_data), unrelated_data, **plot_kwargs)
    elif plot == 'boxplot':
        plt.boxplot([related_data,unrelated_data], **plot_kwargs)
        plt.xticks([1,2],labels=['Related','Unrelated'])
    if title is not None:
        plt.title(title)

    # AUROC
    fpr, tpr, thresholds = roc_curve(['Related']*len(related_data) + ['Unrelated']*len(unrelated_data), list(related_data) + list(unrelated_data), pos_label='Related')
    auroc = auc(fpr, tpr)

    plt.subplot(122)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if title is not None:
        plt.title('ROC - ' + title)

    # Save plot
    if save_path is not None:
        plt.savefig(save_path)


def save_file(data, path, format=None, **savekwargs):
    path = str(path)
    if format is None:
        format = str(path).lower().split('.')[-1]

    #
    # Pre-transform data to other type if needed
    #
    if sparse.issparse(data) and format in ['csv','tsv']:
        data = data.toarray()

    #
    # Define save function and arguments for different data type
    #
    save = {
        'numpy': {
            'csv': {'func': np.savetxt,
                    'kwargs': {'delimiter': ',', 'fname': path, 'X': data}},
            'tsv': {'func': np.savetxt,
                    'kwargs': {'delimiter': '\t', 'fname': path, 'X': data}},
            'npy': {'func': np.save,
                    'kwargs': {'file': path, 'arr': data}},
        },
        'scipy': {
            'npz': {'func': sparse.save_npz,
                    'kwargs': {'file': path, 'matrix': data}},
        },
        'pandas_df': {
            'csv': {'func': lambda df, **kwargs: df.to_csv(**kwargs),
                    'kwargs': {'df': data, 'path_or_buf': path, 'sep': ','}},
            'tsv': {'func': lambda df, **kwargs: df.to_csv(**kwargs),
                    'kwargs': {'df': data, 'path_or_buf': path, 'sep': '\t'}},
            'parquet': {'func': lambda df, **kwargs: df.to_parquet(**kwargs),
                        'kwargs': {'df': data, 'fname': path}},
            'pkl': {'func': lambda df, **kwargs: df.to_pickle(**kwargs),
                    'kwargs': {'df': data, 'path': path}},
        },
        'pandas_series': {
            'csv': {'func': lambda df, **kwargs: df.to_csv(**kwargs),
                    'kwargs': {'df': data, 'path': path, 'sep': ','}},
            'tsv': {'func': lambda df, **kwargs: df.to_csv(**kwargs),
                    'kwargs': {'df': data, 'path': path, 'sep': '\t'}},
            'pkl': {'func': lambda df, **kwargs: df.to_pickle(**kwargs),
                    'kwargs': {'df': data, 'path': path}},
        },
    }

    if isinstance(data, np.ndarray):
        data_type = 'numpy'
    elif sparse.issparse(data):
        data_type = 'scipy'
    elif isinstance(data, pd.DataFrame):
        data_type = 'pandas_df'
    elif isinstance(data, pd.Series):
        data_type = 'pandas_series'
    else:
        data_type = None

    assert format in [key for key in save[data_type]], 'Shoule be one of following format {}'.format([key for key in save[data_type]])
    save[data_type][format]['func'](**save[data_type][format]['kwargs'],**savekwargs)


def read_file(path, data_type=None, format=None, **readkwargs):
    path = str(path)
    assert os.path.isfile(path), '[Error] {} is not a file'.format(path)

    if format is None:
        format = str(path).lower().split('.')[-1]

    if data_type is None:
        if format == 'npy':
            data_type = 'numpy'
        elif format == 'npz':
            data_type = 'scipy'
        elif format == 'parquet':
            data_type = 'pandas_df'
        else:
            data_type = 'pandas_df'

    #
    # Define read function and arguments for different data type
    #
    read = {
        'numpy': {
            'csv': {'func': np.loadtxt,
                    'kwargs': {'delimiter': ',', 'fname': path}},
            'tsv': {'func': np.loadtxt,
                    'kwargs': {'delimiter': '\t', 'fname': path}},
            'npy': {'func': np.load,
                    'kwargs': {'file': path}},
        },
        'scipy': {
            'csv': {'func': lambda **kwargs: sparse.csr_matrix(np.loadtxt(**kwargs)),
                    'kwargs': {'delimiter': ',', 'fname': path}},
            'tsv': {'func': lambda **kwargs: sparse.csr_matrix(np.loadtxt(**kwargs)),
                    'kwargs': {'delimiter': '\t', 'fname': path}},
            'npz': {'func': sparse.load_npz,
                    'kwargs': {'file': path}},
        },
        'pandas_df': {
            # Direct reciprocal of df.to_csv is pandas.DataFrame.from_csv, but from_csv is deprecated.
            # Used read_csv with default of index_col and parse_dates instead
            'csv': {'func': pd.read_csv,
                    'kwargs': {'filepath_or_buffer': path, 'sep': ',', 'index_col': 0, 'parse_dates': True}},
            'tsv': {'func': pd.read_csv,
                    'kwargs': {'filepath_or_buffer': path, 'sep': '\t', 'index_col': 0, 'parse_dates': True}},
            'parquet': {'func': pd.read_parquet,
                        'kwargs': {'path': path}},
            'pkl': {'func': pd.read_pickle,
                    'kwargs': {'path': path}},
        },
        'pandas_series': {
            # Direct reciprocal of series.to_csv is pandas.Series.from_csv, but from_csv is deprecated.
            # Used read_csv with default of index_col and parse_dates and header instead
            'csv': {'func': pd.read_csv,
                    'kwargs': {'filepath_or_buffer': path, 'sep': ',', 'index_col': 0, 'parse_dates': True, 'header': None, 'squeeze': True}},
            'tsv': {'func': pd.read_csv,
                    'kwargs': {'filepath_or_buffer': path, 'sep': '\t', 'index_col': 0, 'parse_dates': True, 'header': None, 'squeeze': True}},
            'pkl': {'func': pd.read_pickle,
                    'kwargs': {'path': path}},
        },
    }

    assert data_type in [_ for _ in read]
    assert format in [key for key in read[data_type]]

    return read[data_type][format]['func'](**read[data_type][format]['kwargs'],**readkwargs)


if __name__ == '__main__':
    import scipy.sparse as sparse
    list_cnt1 = [[1,1,0,1],[0,1,0,1],[0,1,1,1]]
    sparse_cnt1 = sparse.csr_matrix(list_cnt1)
    ndarray_cnt1 = np.array(list_cnt1)
    result_cnt1 = np.array([[0., 0.816496580927726, 0.6666666666666669],[0.816496580927726, 0., 0.816496580927726],[0.6666666666666669, 0.816496580927726, 0.]])

    assert np.array_equal(pairwise_similarity(list_cnt1), result_cnt1)
    assert np.array_equal(pairwise_similarity(sparse_cnt1), result_cnt1)
    assert np.array_equal(pairwise_similarity(ndarray_cnt1), result_cnt1)

