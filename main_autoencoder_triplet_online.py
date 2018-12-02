import tensorflow as tf, os, scipy.sparse as sparse, pandas as pd, numpy as np, joblib, logging
from autoencoder.autoencoder_triplet import DenoisingAutoencoder
from autoencoder.autoencoder_triplet_online import DenoisingAutoencoderTripletOnline
import datasets.articles as articles
import helpers
from pathlib import Path

# Define path
_script_path = Path(os.path.dirname(os.path.realpath(__file__)))


# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_boolean('verbose', False, 'Level of verbosity. 0 - silent, 1 - print log')
flags.DEFINE_boolean('encode_full', False, 'Whether to encode and store the full data set')

# Count Vectorizer parameters
flags.DEFINE_boolean('restore_previous_data', False, 'If true, restore previous data corresponding to model name')
flags.DEFINE_float('min_df', 0, 'min_df for sklearn CountVectorizer')
flags.DEFINE_float('max_df', 0.99, 'max_df for sklearn CountVectorizer')
flags.DEFINE_integer('max_features', 10000, 'max_features for sklearn CountVectorizer')

# Stacked Denoising Autoencoder specific parameters
flags.DEFINE_string('model_name', '', 'Model name.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters')
flags.DEFINE_integer('compress_factor', 20, 'Compression factor to determine num. of hidder nodes')
flags.DEFINE_string('corr_type', 'masking', 'Type of input corruption. ["none", "masking", "salt_and_pepper", "decay]')
flags.DEFINE_float('corr_frac', 0.3, 'Fraction of the input to corrupt.')
flags.DEFINE_integer('xavier_init', 1, 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('enc_act_func', 'sigmoid', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dec_act_func', 'sigmoid', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('main_dir', '', 'Directory to store data relative to the algorithm. Same as model_name if empty')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs, set to 0 will not train the model')
flags.DEFINE_integer('batch_size', 100, 'Size of each mini-batch.') #todo: allows batch_size to be set 0-1
flags.DEFINE_float('alpha', 0.01, 'hyper parameter for balancing similarity in loss function')

assert 0. <= FLAGS.min_df <= 1.
assert 0. <= FLAGS.max_df <= 1.
assert FLAGS.max_features >= 1
assert FLAGS.enc_act_func in ['sigmoid', 'tanh']
assert FLAGS.dec_act_func in ['sigmoid', 'tanh', 'none']
assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'decay', 'none']
assert 0. <= FLAGS.corr_frac <= 1.
assert FLAGS.loss_func in ['cross_entropy', 'mean_squared']
assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum']

if FLAGS.main_dir == '': FLAGS.main_dir = FLAGS.model_name


# ############## #
#   Set logger   #
# ############## #
# todo: set logger
# FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
# LOG_FILE = "my_app.log"
#
# def get_console_handler():
#    console_handler = logging.StreamHandler(sys.stdout)
#    console_handler.setFormatter(FORMATTER)
#    return console_handler
# def get_file_handler():
#    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
#    file_handler.setFormatter(FORMATTER)
#    return file_handler
# def get_logger(logger_name):
#    logger = logging.getLogger(logger_name)
#    logger.setLevel(logging.DEBUG) # better to have too much log than not enough
#    logger.addHandler(get_console_handler())
#    logger.addHandler(get_file_handler())
#    # with this pattern, it's rarely necessary to propagate the error up to parent
#    logger.propagate = False
#    return logger


if __name__ == '__main__':
    print(__file__ + ': Start')

    # init the model
    model = DenoisingAutoencoderTripletOnline(
        seed=FLAGS.seed, model_name=FLAGS.model_name, compress_factor=FLAGS.compress_factor,
        enc_act_func=FLAGS.enc_act_func, dec_act_func=FLAGS.dec_act_func, xavier_init=FLAGS.xavier_init,
        corr_type=FLAGS.corr_type, corr_frac=FLAGS.corr_frac,
        loss_func=FLAGS.loss_func, main_dir=FLAGS.main_dir, opt=FLAGS.opt,
        learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum,
        verbose=FLAGS.verbose, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
        alpha=FLAGS.alpha)

    # Prepare data
    if FLAGS.restore_previous_data:
        article_contents = pd.read_parquet(model.data_dir + 'article_contents.snappy.parquet')
        X = sparse.load_npz(model.data_dir + 'article_contents_binary_count_vectorized.npz')
        X_pos = sparse.load_npz(model.data_dir + 'article_contents_binary_count_vectorized_pos.npz')
        X_neg = sparse.load_npz(model.data_dir + 'article_contents_binary_count_vectorized_neg.npz')
        X_tfidf = sparse.load_npz(model.data_dir + 'article_contents_tfidf_vectorized.npz')
        count_vectorizer = joblib.load(model.data_dir + 'count_vectorizer.joblib')
        tfidf_transformer = joblib.load(model.data_dir + 'tfidf_transformer.joblib')
    else:
        article_contents = articles.read_articles(path='/Users/user/Documents/hk01/cache/s3/article_contents/latest.snappy.parquet')
        #article_contents = articles.similar_articles(article_contents, id_colname='article_id', cate_colname='title_group', min_cate=2, max_cate=20)

        title_group_value_counts = article_contents.title_group.value_counts()
        indices = article_contents.title_group.isin(title_group_value_counts[(title_group_value_counts>=2) & (title_group_value_counts<=100)].index)

        article_contents['group'] = ''
        article_contents.loc[indices,'group'] = article_contents[indices]['title_group']
        article_contents.loc[~indices, 'group'] = article_contents[~indices]['main_category_id']
        article_contents['label'] = pd.factorize(article_contents.group)[0] + 1

        row = 1000
        count_vectorizer, X, X_pos, X_neg = articles.count_vectorize(
            article_contents.main_content[0:row],
            #article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_pos[0:row]],
            #article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_pos[0:row]],
            min_df=FLAGS.min_df,
            max_df=FLAGS.max_df,
            max_features=FLAGS.max_features,
            binary=False
        )
        tfidf_transformer, X_tfidf = articles.tfidf_transform(X)

        # Save training data
        #article_contents.to_parquet(model.data_dir + 'article_contents.snappy.parquet') #todo: urgent need to be fixed !!!!!!!!!!!!!
        sparse.save_npz(model.data_dir + 'article_contents_count_vectorized.npz', X)
        X.data = np.array([1] * len(X.data))
        sparse.save_npz(model.data_dir + 'article_contents_binary_count_vectorized.npz', X)
        if X_pos is not None: sparse.save_npz(model.data_dir + 'article_contents_binary_count_vectorized_pos.npz', X_pos)
        if X_neg is not None: sparse.save_npz(model.data_dir + 'article_contents_binary_count_vectorized_neg.npz', X_neg)
        sparse.save_npz(model.data_dir + 'article_contents_tfidf_vectorized.npz', X_tfidf)

        # Save vectorizer
        joblib.dump(count_vectorizer, model.data_dir + 'count_vectorizer.joblib')
        joblib.dump(tfidf_transformer, model.data_dir + 'tfidf_transformer.joblib')

    # trX = {'org': X[:-100],
    #        'pos': X_pos[:-100],
    #        'neg': X_neg[:-100]}
    # vlX = {'org': X[-100:],
    #        'pos': X_pos[-100:],
    #        'neg': X_neg[-100:]}
    # teX = None

    trX = X[:-100]
    trX_label = article_contents.label[0:row][:-100]
    vlX = X[-100:]
    vlX_label = article_contents.label[0:row][-100:]
    teX=None

    # Fit the model
    model.fit(trX, trX_label, vlX, vlX_label, restore_previous_model=FLAGS.restore_previous_model)

    # Encode the data and store it
    X_encoded = model.transform(X, name='full', save=FLAGS.encode_full)

    # Print top 10 similar articles
    article_binary_count_cosine_sim = helpers.pairwise_similarity(X, metric='cosine')
    article_tfidf_cosine_sim = helpers.pairwise_similarity(X_tfidf, metric='linear kernel') #This is same as cosine similarity as X_tfidf is l2 normalized (refer to sklearn's TFIDFTransformer for this)
    article_embedding_cosine_sim = helpers.pairwise_similarity(X_encoded, metric='cosine')
    for i,v in enumerate(np.nanargmax(article_embedding_cosine_sim,1)[0:10]):
        print(article_contents[['category_publish_name','title']].iloc[i])
        print('most similar article using count vectorizer')
        print(article_contents[['category_publish_name', 'title']].iloc[np.nanargmax(article_binary_count_cosine_sim,1)[i]])
        print('most similar article using DAE')
        print(article_contents[['category_publish_name','title']].iloc[v])
        print('score: {}'.format(article_embedding_cosine_sim[i,v]))
        print()

    print(__file__ + ': End')


