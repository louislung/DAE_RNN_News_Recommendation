from ..helpers import save_file
from ..helpers import read_file
import numpy as np, pandas as pd, scipy.sparse as sparse, os
from pathlib import Path

_script_path = Path(os.path.dirname(os.path.realpath(__file__)))

def test_save_read_file():
    # Test numpy array
    nparray = []
    nparray.append(np.array([0, 2, 3, 4, ]))
    nparray.append(np.array([[0, 2], [2.3, 0]]))

    for data in nparray:
        for file in ['test.csv', 'test.tsv', 'test.npy']:
            save_file(data, path=_script_path / file)
            data_read = read_file(_script_path / file, data_type='numpy')
            assert (data == data_read).all(), (data, data_read)
            os.remove(_script_path / file)

    # Test scipy sparse matrix
    sp = []
    sp.append(sparse.csr_matrix([0, 0, 0, 0]))
    sp.append(sparse.csr_matrix([[0, 0, 0, 0], [0, 0, 0, 0]]))
    sp.append(sparse.csr_matrix([1, 2.2, 0, 0]))
    sp.append(sparse.csr_matrix([[1, 2.2, 0, 0], [1, 2.2, 5.12312313, 0]]))

    for data in sp:
        for file in ['test.csv', 'test.tsv', 'test.npz']:
            save_file(data, path=_script_path / file)
            data_read = read_file(_script_path / file, data_type='scipy')
            assert (data != data_read).nnz == 0
            os.remove(_script_path / file)

    # Test pandas dataframe
    pd_df = []
    pd_df.append(pd.DataFrame([0, 1, 2], index=[5, 3, 2], columns=['dummy']))
    pd_df.append(pd.DataFrame([[0, 1, 2],[2,3,4]], index=['apple','boy'], columns=['dummy','d2','d3']))
    pd_df.append(pd.DataFrame(['apple','boy','cat'], index=[5, 3, 2], columns=['dummy']))
    pd_df.append(pd.DataFrame([['apple','boy','cat'], ['apple1','boy1','cat1']], index=['apple', 'boy'], columns=['dummy', 'd2', 'd3']))

    for data in pd_df:
        for file in ['test.csv', 'test.tsv', 'test.parquet', 'test.pkl']:
            save_file(data, path=_script_path / file)
            data_read = read_file(_script_path / file, data_type='pandas_df')
            assert data.equals(data_read)
            os.remove(_script_path / file)

    # Test pandas series
    pd_series = []
    pd_series.append(pd.Series([0, 1, 2], index=[5,4,3]))
    pd_series.append(pd.Series(['a', 'b', 'c']))

    for data in pd_series:
        for file in ['test.csv', 'test.tsv', 'test.pkl']:
            save_file(data, path=_script_path / file)
            data_read = read_file(_script_path / file, data_type='pandas_series')
            print(data)
            print(data_read)
            assert data.equals(data_read)
            os.remove(_script_path / file)