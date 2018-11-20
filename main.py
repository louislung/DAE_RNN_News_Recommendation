from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import os, sys, jieba, pandas as pd, numpy as np, tensorflow as tf
import autoencoder
import autoencoder_triplet

# ############# #
#   Parameter   #
# ############# #
param_count_vectorizer = {
    'max_df': 0.99,
    'min_df': 0.01,
    'max_features': 10000
}


# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('model', '', 'Which model to use. ["dae", "dae_triplet"]')
flags.DEFINE_string('model_name', '', 'Model name.')
flags.DEFINE_string('dataset', 'article', 'Which dataset to use. ["mnist", "cifar10", "article"]')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_boolean('encode_train', False, 'Whether to encode and store the training set.')
flags.DEFINE_boolean('encode_valid', False, 'Whether to encode and store the validation set.')
flags.DEFINE_boolean('encode_test', False, 'Whether to encode and store the test set.')


# Stacked Denoising Autoencoder specific parameters
flags.DEFINE_integer('n_components', 256, 'Number of hidden units in the dae.')
flags.DEFINE_string('corr_type', 'none', 'Type of input corruption. ["none", "masking", "salt_and_pepper", "decay]')
flags.DEFINE_float('corr_frac', 0., 'Fraction of the input to corrupt.')
flags.DEFINE_integer('xavier_init', 1, 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('enc_act_func', 'sigmoid', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dec_act_func', 'sigmoid', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('main_dir', 'dae/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_integer('weight_images', 0, 'Number of weight images to generate.')
flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
flags.DEFINE_integer('num_epochs', 500, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 100, 'Size of each mini-batch.')
flags.DEFINE_float('alpha', 1, 'hyper parameter for balancing similarity in loss function')

assert FLAGS.dataset in ['mnist', 'cifar10', 'article']
assert FLAGS.enc_act_func in ['sigmoid', 'tanh']
assert FLAGS.dec_act_func in ['sigmoid', 'tanh', 'none']
assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'decay', 'none']
assert 0. <= FLAGS.corr_frac <= 1.
assert FLAGS.loss_func in ['cross_entropy', 'mean_squared']
assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum']


# ######################## #
#   Prepare article data   #
# ######################## #
def find_positive_item(input_id, pos=True):
    """ Find positive (similar) or negative (dissimilar) item by id.

    :param id: int, index.
    :param pos: bool, return positive item's id if True else return negative item's id.

    :return: another id
    """

    id_list = article_contents[~article_contents.index.isin([input_id])][article_contents.main_category_id == article_contents.main_category_id.loc[input_id]].article_id

    return min(id_list, key = lambda x: abs(x-input_id))

def tokenizer_custom(text):
    return [word for word in jieba.cut(text) if len(word)>1 and not word.isdigit()] #todo: need to take care 5%, 5.5, etc

article_contents = pd.read_parquet('/Users/user/Documents/hk01/cache/s3/article_contents/latest.snappy.parquet')
article_contents.index = article_contents.article_id
article_contents = article_contents[article_contents.main_content.str.strip() != '']
article_contents = article_contents[article_contents.main_content.notna()]

article_contents['article_id_pos'] = 0
for cate_id in article_contents.main_category_id.unique():
    article_contents = pd.merge(article_contents,
                                article_contents[article_contents.main_category_id == cate_id][['article_id']].shift(-1).add_suffix('2'),
                                how='left',left_index=True, right_index=True)
    article_contents.loc[article_contents.article_id2.notnull(),'article_id_pos'] = article_contents.article_id2[article_contents.article_id2.notnull()].astype(int)
    article_contents.drop(columns=['article_id2'], inplace=True)

article_contents['article_id_neg'] = 0
article_contents['article_id_neg'] = list(article_contents.article_id.sample(frac=1))

article_contents['valid_triplet_data'] = 0
article_contents.loc[(article_contents.article_id_pos != 0) & (article_contents.article_id_neg != 0),'valid_triplet_data'] = 1

article_contents.to_parquet('data/article_contents_processed.snappy.parquet')
#article_contents = article_contents.assign(article_id_pos = article_contents.apply(lambda row: find_positive_item(row.name),axis=1))

count_vectorizer = CountVectorizer(tokenizer = tokenizer_custom, **param_count_vectorizer)
X = count_vectorizer.fit_transform(article_contents[article_contents.valid_triplet_data == 1].main_content[0:1000])
X_pos = count_vectorizer.transform(article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_pos[0:1000]])
X_neg = count_vectorizer.transform(article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_neg[0:1000]])



if __name__ == '__main__':

    if FLAGS.dataset == 'mnist':

        # ################# #
        #   MNIST Dataset   #
        # ################# #

        trX, vlX, teX = datasets.load_mnist_dataset(mode='unsupervised')

    elif FLAGS.dataset == 'cifar10':

        # ################### #
        #   Cifar10 Dataset   #
        # ################### #

        trX, teX = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='unsupervised')
        vlX = teX[:5000]  # Validation set is the first half of the test set

    else:  # cannot be reached, just for completeness
        trX = X[:-100]
        vlX = X[-100:]
        teX = None

    # Create the object
    if FLAGS.model == 'dae':
        model = autoencoder.DenoisingAutoencoder(
            seed=FLAGS.seed, model_name=FLAGS.model_name, n_components=FLAGS.n_components,
            enc_act_func=FLAGS.enc_act_func, dec_act_func=FLAGS.dec_act_func, xavier_init=FLAGS.xavier_init,
            corr_type=FLAGS.corr_type, corr_frac=FLAGS.corr_frac, dataset=FLAGS.dataset,
            loss_func=FLAGS.loss_func, main_dir=FLAGS.main_dir, opt=FLAGS.opt,
            learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum,
            verbose=FLAGS.verbose, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size)
    elif FLAGS.model == 'dae_triplet':
        model = autoencoder_triplet.DenoisingAutoencoderTriplet(
            seed=FLAGS.seed, model_name=FLAGS.model_name, n_components=FLAGS.n_components,
            enc_act_func=FLAGS.enc_act_func, dec_act_func=FLAGS.dec_act_func, xavier_init=FLAGS.xavier_init,
            corr_type=FLAGS.corr_type, corr_frac=FLAGS.corr_frac, dataset=FLAGS.dataset,
            loss_func=FLAGS.loss_func, main_dir=FLAGS.main_dir, opt=FLAGS.opt,
            learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum,
            verbose=FLAGS.verbose, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
            alpha=FLAGS.alpha)
        trX = {'org': X[:-100],
               'pos': X_pos[:-100],
               'neg': X_neg[:-100]}
        vlX = {'org': X[-100:],
               'pos': X_pos[-100:],
               'neg': X_neg[-100:]}
        teX = None
    else:
        model = None

    # Fit the model
    model.fit(trX, vlX, restore_previous_model=FLAGS.restore_previous_model)

    # Encode the training data and store it
    model.transform(trX['org'], name='train', save=FLAGS.encode_train)
    model.transform(vlX['pos'], name='validation', save=FLAGS.encode_valid)
    #model.transform(teX, name='test', save=FLAGS.encode_test)

    # save images
    #model.get_weights_as_images(28, 28, max_images=FLAGS.weight_images)

