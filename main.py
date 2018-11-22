import tensorflow as tf, os, scipy.sparse as sparse, pandas as pd
import autoencoder.autoencoder_triplet as autoencoder_triplet
import datasets.articles as articles
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
flags.DEFINE_float('min_df', 0.01, 'min_df for sklearn CountVectorizer')
flags.DEFINE_float('max_df', 0.99, 'max_df for sklearn CountVectorizer')
flags.DEFINE_integer('max_features', 10000, 'max_features for sklearn CountVectorizer')

# Stacked Denoising Autoencoder specific parameters
flags.DEFINE_string('model_name', '', 'Model name.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters')
flags.DEFINE_integer('compress_factor', 10, 'Compression factor to determine num. of hidder nodes')
flags.DEFINE_string('corr_type', 'none', 'Type of input corruption. ["none", "masking", "salt_and_pepper", "decay]')
flags.DEFINE_float('corr_frac', 0., 'Fraction of the input to corrupt.')
flags.DEFINE_integer('xavier_init', 1, 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('enc_act_func', 'sigmoid', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dec_act_func', 'sigmoid', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('main_dir', '', 'Directory to store data relative to the algorithm. Same as model_name if empty')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter')
flags.DEFINE_integer('num_epochs', 500, 'Number of epochs, set to 0 will not train the model')
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


if __name__ == '__main__':
    print(__file__ + ': Start')

    # init the model
    model = autoencoder_triplet.DenoisingAutoencoderTriplet(
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
        X = sparse.load_npz(model.data_dir + 'article_contents_vectorized.npz')
        X_pos = sparse.load_npz(model.data_dir + 'article_contents_vectorized_pos.npz')
        X_neg = sparse.load_npz(model.data_dir + 'article_contents_vectorized_neg.npz')
    else:
        article_contents = articles.read_articles(path='/Users/user/Documents/hk01/cache/s3/article_contents/latest.snappy.parquet',save_path=None,id_colname='article_id',cate_colname='main_category_id')
        row = 1000
        count_vectorizer, X, X_pos, X_neg = articles.count_vectorize(
            article_contents[article_contents.valid_triplet_data == 1].main_content[0:row],
            article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_pos[0:row]],
            article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_pos[0:row]],
            min_df=FLAGS.min_df,
            max_df=FLAGS.max_df,
            max_features=FLAGS.max_features,
        )

    trX = {'org': X[:-100],
           'pos': X_pos[:-100],
           'neg': X_neg[:-100]}
    vlX = {'org': X[-100:],
           'pos': X_pos[-100:],
           'neg': X_neg[-100:]}
    teX = None

    # Fit the model
    model.fit(trX, vlX, restore_previous_model=FLAGS.restore_previous_model)

    # Save training data
    article_contents.to_parquet(model.data_dir + 'article_contents.snappy.parquet')
    sparse.save_npz(model.data_dir + 'article_contents_vectorized.npz', X)
    sparse.save_npz(model.data_dir + 'article_contents_vectorized_pos.npz', X_pos)
    sparse.save_npz(model.data_dir + 'article_contents_vectorized_neg.npz', X_neg)

    # Encode the data and store it
    X_encoded = model.transform(X, name='full', save=FLAGS.encode_full)

    # Print top 10 similar articles
    article_similarity = X_encoded.dot(X_encoded.transpose())
    for i,v in enumerate(article_similarity.argmax(1)[0:10]):
        print(article_contents[article_contents.valid_triplet_data == 1][['category_publish_name','title']].iloc[i])
        print(article_contents[article_contents.valid_triplet_data == 1][['category_publish_name','title']].iloc[v])
        print()


    print(__file__ + ': End')


