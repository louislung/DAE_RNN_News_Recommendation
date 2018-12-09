import tensorflow as tf, os, scipy.sparse as sparse, pandas as pd, numpy as np, joblib, logging
from autoencoder.autoencoder_triplet import DenoisingAutoencoder
from autoencoder.autoencoder_triplet_online import DenoisingAutoencoderTripletOnline
import datasets.articles as articles
import helpers
from autoencoder import utils
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
flags.DEFINE_integer('verbose_step', 5, 'Print log every x training steps')
flags.DEFINE_boolean('encode_full', False, 'Whether to encode and store the full data set')
flags.DEFINE_boolean('validation', False, 'Whether to use a validation set and print validation loss')
flags.DEFINE_string('input_format', 'count', 'Input data format. ["binary", "tfidf"]')

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
flags.DEFINE_string('triplet_strategy', 'batch_all', 'triplet strategy ["batch_all","batch_hard"]')

assert 0. <= FLAGS.min_df <= 1.
assert 0. <= FLAGS.max_df <= 1.
assert FLAGS.max_features >= 1
assert FLAGS.enc_act_func in ['sigmoid', 'tanh']
assert FLAGS.dec_act_func in ['sigmoid', 'tanh', 'none']
assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'decay', 'none']
assert 0. <= FLAGS.corr_frac <= 1.
assert FLAGS.loss_func in ['cross_entropy', 'mean_squared']
assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum']
assert FLAGS.verbose_step > 0
assert FLAGS.triplet_strategy in ['batch_all','batch_hard']
assert FLAGS.input_format in ['binary','tfidf']

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
        verbose=FLAGS.verbose, verbose_step=FLAGS.verbose_step, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
        alpha=FLAGS.alpha, triplet_strategy=FLAGS.triplet_strategy)

    # set row
    train_row = 40000
    validate_row = 5000

    # write parameter
    with open(model.parameter_file, 'a+') as text_file:
        print('train_row={}'.format(train_row), file=text_file)
        print('validate_row={}'.format(validate_row), file=text_file)
        print('input_format={}'.format(FLAGS.input_format), file=text_file)

    # prepare or restore data
    if FLAGS.restore_previous_data:
        article_contents = pd.read_parquet(model.data_dir + 'article.snappy.parquet')
        article_contents.append(pd.read_parquet(model.data_dir + 'article_validate.snappy.parquet'))
        X = sparse.load_npz(model.data_dir + 'article_binary_count_vectorized.npz')
        X_validate = sparse.load_npz(model.data_dir + 'article_binary_count_vectorized_validate.npz')
        X_label = pd.read_pickle(model.data_dir + 'article_label.pkl')
        X_label_validate = pd.read_pickle(model.data_dir + 'article_label_validate.pkl')
        X_tfidf = sparse.load_npz(model.data_dir + 'article_tfidf_vectorized.npz')
        X_tfidf_validate = sparse.load_npz(model.data_dir + 'article_tfidf_vectorized_validate.npz')
        count_vectorizer = joblib.load(model.data_dir + 'count_vectorizer.joblib')
        tfidf_transformer = joblib.load(model.data_dir + 'tfidf_transformer.joblib')
    else:
        article_contents = articles.read_articles(path='/Users/user/Documents/hk01/cache/s3/article_contents/latest.snappy.parquet')

        title_group_value_counts = article_contents.title_group.value_counts()
        indices = article_contents.title_group.isin(title_group_value_counts[(title_group_value_counts>=2) & (title_group_value_counts<=100)].index)

        article_contents['group'] = ''
        article_contents.loc[indices,'group'] = article_contents[indices]['title_group']
        article_contents.loc[~indices, 'group'] = article_contents[~indices]['category_publish_name']
        article_contents = article_contents.loc[indices,]
        article_contents['label'] = pd.factorize(article_contents.group)[0] + 1

        #article_contents['label'] = pd.factorize(article_contents.category_publish_name)[0] + 1

        X_label = article_contents.label[0:train_row]
        count_vectorizer, X, X_pos, X_neg = articles.count_vectorize(
            article_contents.main_content[0:train_row],
            min_df=FLAGS.min_df,
            max_df=FLAGS.max_df,
            max_features=FLAGS.max_features,
            binary=False
        )
        tfidf_transformer, X_tfidf = articles.tfidf_transform(X)

        X_validate = count_vectorizer.transform(article_contents.main_content[train_row:validate_row+train_row])
        X_label_validate = article_contents.label[train_row:validate_row+train_row]
        X_tfidf_validate = tfidf_transformer.transform(X_validate)

        # Save training & validation data
        article_contents.iloc[0:train_row,].to_parquet(model.data_dir + 'article.snappy.parquet')
        article_contents.iloc[train_row:validate_row+train_row,].to_parquet(model.data_dir + 'article_validate.snappy.parquet')
        X_label.to_pickle(model.data_dir + 'article_label.pkl')
        X_label_validate.to_pickle(model.data_dir + 'article_label_validate.pkl')
        sparse.save_npz(model.data_dir + 'article_count_vectorized.npz', X)
        sparse.save_npz(model.data_dir + 'article_count_vectorized_validate.npz', X_validate)
        X.data = np.array([1] * len(X.data))
        X_validate.data = np.array([1] * len(X_validate.data))
        sparse.save_npz(model.data_dir + 'article_binary_count_vectorized.npz', X)
        sparse.save_npz(model.data_dir + 'article_binary_count_vectorized_validate.npz', X_validate)
        sparse.save_npz(model.data_dir + 'article_tfidf_vectorized.npz', X_tfidf)
        sparse.save_npz(model.data_dir + 'article_tfidf_vectorized_validate.npz', X_tfidf_validate)

        # Save in tsv format for visualization in tensorboard (http://projector.tensorflow.org/)
        np.savetxt(model.data_dir + 'tsv/' + 'article_tfidf_vectorized.tsv',  X_tfidf.toarray(), delimiter='\t')
        np.savetxt(model.data_dir + 'tsv/' + 'article_tfidf_vectorized_validate.tsv', X_tfidf_validate.toarray(), delimiter='\t')
        np.savetxt(model.data_dir + 'tsv/' + 'article_binary_count_vectorized.tsv', X.toarray(), delimiter='\t')
        np.savetxt(model.data_dir + 'tsv/' + 'article_binary_count_vectorized_validate.tsv', X_validate.toarray(), delimiter='\t')
        article_contents.iloc[0:train_row,][['label', 'title', 'title_group', 'group', 'category_publish_name']].to_csv(model.data_dir + 'article_label.tsv',  sep='\t')
        article_contents.iloc[train_row:validate_row+train_row,][['label', 'title', 'title_group', 'group', 'category_publish_name']].to_csv(model.data_dir + 'article_label_validate.tsv', sep='\t')

        # Save vectorizer
        joblib.dump(count_vectorizer, model.data_dir + 'count_vectorizer.joblib')
        joblib.dump(tfidf_transformer, model.data_dir + 'tfidf_transformer.joblib')

    data_dict = {
        'binary': {
            'train': X,
            'validate': X_validate,
        },
        'tfidf': {
            'train': X_tfidf,
            'validate': X_tfidf_validate,
        }
    }

    trX = data_dict[FLAGS.input_format]['train']
    trX_label = X_label
    vlX = None
    vlX_label = None
    if FLAGS.validation:
        vlX = data_dict[FLAGS.input_format]['validate']
        vlX_label = X_label_validate
    teX=None

    # Fit the model
    print('fit')
    model.fit(trX, trX_label, vlX, vlX_label, restore_previous_model=FLAGS.restore_previous_model)
    print('fit done')

    # Encode the data and store it
    X_encoded = model.transform(utils.decay_noise(X_tfidf, FLAGS.corr_frac), name='full', save=FLAGS.encode_full)
    np.savetxt(model.data_dir + 'article_encoded.tsv', X_encoded, delimiter='\t')
    X_encoded_validate = model.transform(utils.decay_noise(X_tfidf_validate, FLAGS.corr_frac), name='full', save=FLAGS.encode_full)
    np.savetxt(model.data_dir + 'article_encoded_validate.tsv', X_encoded_validate, delimiter='\t')

    # Print top 10 similar articles
    article_binary_count_cosine_sim = helpers.pairwise_similarity(X, metric='cosine')
    article_tfidf_cosine_sim = helpers.pairwise_similarity(X_tfidf, metric='linear kernel') #This is same as cosine similarity as X_tfidf is l2 normalized (refer to sklearn's TFIDFTransformer for this)
    article_embedding_cosine_sim = helpers.pairwise_similarity(X_encoded, metric='cosine')

    article_binary_count_cosine_sim_argmax = np.nanargmax(article_binary_count_cosine_sim, 1)
    for i,v in enumerate(np.nanargmax(article_embedding_cosine_sim,1)[0:5]):
        print(article_contents[['category_publish_name','title']].iloc[i])
        print('most similar article using count vectorizer')
        print(article_contents[['category_publish_name', 'title']].iloc[article_binary_count_cosine_sim_argmax[i]])
        print('most similar article using DAE')
        print(article_contents[['category_publish_name','title']].iloc[v])
        print('score: {}'.format(article_embedding_cosine_sim[i,v]))
        print()

    print(__file__ + ': End')


