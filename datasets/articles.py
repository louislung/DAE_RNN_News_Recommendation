import os, pandas as pd, jieba, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# Define path
_script_path = Path(os.path.dirname(os.path.realpath(__file__)))


# ################################### #
#   Fn for read and process article   #
# ################################### #
def find_positive_item(input_id, pos=True, cate_colname='main_category_id'):
    """ Find positive (similar) or negative (dissimilar) item by id.

    :param id: index (same as aritcle id)
    :param pos: return positive item's id if True else return negative item's id

    :type id: int
    :type pos: bool

    :return: another id
    :rtype: int

    .. note:: article_contents = article_contents.assign(article_id_pos = article_contents.apply(lambda row: find_positive_item(row.name),axis=1))
    """
    id_list = article_contents[~article_contents.index.isin([input_id])][article_contents[cate_colname] == article_contents[cate_colname].loc[input_id]].article_id

    return min(id_list, key = lambda x: abs(x-input_id))


def tokenizer_chinese(text):
    """ Custom tokenizer to be used in sklean

    :param text: path of parquet to be read

    :type text: str

    :return: tokenized word
    :rtype: list of str
    """
    # todo: need to take care 5%, 5.5, etc
    # todo: word in bracket should consider as one word? e.g.「磁力王」,（Michael Fassbender）, 《蝙蝠俠對超人：正義曙光》,（男星Ed Skrein飾演）
    return [word for word in jieba.cut(text) if len(word)>1 and not word.isdigit()]


def read_articles(path='/Users/user/Documents/hk01/cache/s3/article_contents/latest.snappy.parquet'):
    """ Read articles data

    :param path: path of parquet to be read

    :type path: str or pathlib.PosixPath

    :return: processed data with extra columns: title_group
    :rtype: pandas.core.frame.DataFrame

    .. note:: the id of input data should be in numeric format
    """
    out_df = pd.read_parquet(path)
    out_df.index = out_df.article_id
    out_df = out_df[out_df.main_content.str.strip() != '']
    out_df = out_df[out_df.main_content.notna()]

    # Add column based on title, e.g. extract 食物設計 from 【食物設計（下）】
    out_df['title_group'] = out_df.title.str.extract('【(.*?)[（|】]')

    return out_df


def save_articles(in_df, save_path = 'data/article_contents_processed.snappy.parquet'):
    """ Read articles data and map a positive and a negative article for every article

    :param in_df: data to be saved

    :type in_df: pandas.core.frame.DataFrame
    """

    in_df.to_parquet(save_path)
    print('Data saved to {}'.format(save_path))


def similar_articles(out_df, id_colname='article_id', cate_colname='main_category_id', min_cate=2, max_cate=None):
    """ Read articles data and map a positive and a negative article for every article

    :param out_df: data
    :param id_colname: column name of the id column
    :param cate_colname: column name of the category column
    :param min_cate: only value in cate_colname which contains >= min_cate articles will be considered
    :param max_cate: only value in cate_colname which contains <= max_cate articles will be considered

    :type path: pandas.core.frame.DataFrame
    :type id_colname: str
    :type cate_colname: str
    :type min_cate: int
    :type max_cate: int or None

    :return: processed data with extra columns: xxx_pos, xxx_neg, valid_triplet_data
    :rtype: pandas.core.frame.DataFrame

    .. note:: the id of input data should be in numeric format
    """

    id_pos_colname = id_colname + '_pos'
    id_neg_colname = id_colname + '_neg'
    cate_value_counts = out_df[cate_colname].value_counts()
    cate_value_counts = cate_value_counts[(cate_value_counts <= (np.inf if max_cate is None else max_cate)) & (cate_value_counts >= min_cate)]

    # For each record, find another record under same category as positive item
    out_df[id_pos_colname] = 0
    out_df[id_neg_colname] = 0
    for cate_id in cate_value_counts.index:
        out_df = pd.merge(out_df,
                          out_df[out_df[cate_colname] == cate_id][[id_colname]].shift(-1).add_suffix('2'),
                          how='left',left_index=True, right_index=True)
        indices = out_df[id_colname + '2'].notnull()
        out_df.loc[indices, id_pos_colname] = out_df[id_colname + '2'][indices].astype(int).tolist()
        out_df.loc[indices, id_neg_colname] = out_df[out_df[cate_colname] != cate_id][id_colname].sample(np.count_nonzero(indices)).tolist()
        out_df.drop(columns=[id_colname + '2'], inplace=True)

    # For each record, find a random id as negative item
    # out_df[id_neg_colname] = list(out_df[id_colname].sample(frac=1))

    # valid_triplet_data = 1 only if the record has both pos and neg item
    out_df['valid_triplet_data'] = 0
    out_df.loc[(out_df[id_pos_colname] != 0) & (out_df[id_pos_colname].notnull()) & (out_df[id_neg_colname] != 0) & (out_df[id_neg_colname].notnull()) ,'valid_triplet_data'] = 1

    return out_df


def count_vectorize(in_series, in_pos_series=None, in_neg_series=None, tokenizer=tokenizer_chinese, **param_count_vectorizer):
    """ Use sklearn's CountVectorizer to fit transform input data

    :param in_series: data to be fit and transformed
    :param in_pos_series: data to be  transformed, each data should corresponding to a positive item of in_series
    :param in_neg_series: data to be transformed, each data should corresponding to a negative item of in_series
    :param tokenizer: tokenizer to be used in CountVectorizer
    :param **param_count_vectorizer: extra parameter for CountVectorizer, e.g. min_df, max_df, max_features

    :type in_series: pandas.Series
    :type in_pos_series: pandas.Series
    :type in_neg_series: pandas.Series

    :return: CountVectorizer and output from CountVectorizer
    :rtype: csr_matrix
    """
    count_vectorizer = CountVectorizer(tokenizer=tokenizer, **param_count_vectorizer) #todo: allows to restore count vectorizer

    X = count_vectorizer.fit_transform(in_series)
    # Only do transform for in_pos_series and in_neg_series so all output have same number of features (columns)
    X_pos = None if in_pos_series is None else count_vectorizer.transform(in_pos_series)
    X_neg = None if in_neg_series is None else count_vectorizer.transform(in_neg_series)

    if in_pos_series is not None: assert X.shape[1] == X_pos.shape[1]
    if in_neg_series is not None: assert X.shape[1] == X_neg.shape[1]

    return count_vectorizer, X, X_pos, X_neg


def tfidf_transform(in_df, **param_tfidf_transformer):
    """ Use sklearn's CountVectorizer to fit transform input data

    :param in_df: matrix of word count to be fit and transformed
    :param **param_tfidf_transformer: extra parameter for TfidfTransformer, e.g. norm, use_idf, smooth_idf, sublinear_tf

    :type in_df: scipy sparse matrix

    :return: TfidfTransformer and output from TfidfTransformer
    :rtype: csr_matrix
    """
    tfidf_transformer = TfidfTransformer(**param_tfidf_transformer)
    X = tfidf_transformer.fit_transform(in_df)

    return tfidf_transformer, X


if __name__ == '__main__':
    article_contents = read_articles(save_path=None)
    row = 1000
    count_vectorizer, X, X_pos, X_neg = count_vectorize(article_contents[article_contents.valid_triplet_data == 1].main_content[0:row],
                                                        article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_pos[0:row]],
                                                        article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_pos[0:row]],
                                                        min_df=0.01,
                                                        max_df=0.99,
                                                        max_features=1000
                                                        )
    tfidf_transformer, X_tfidf = tfidf_transform(X)