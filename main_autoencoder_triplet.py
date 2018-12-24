import tensorflow as tf, os, scipy.sparse as sparse, pandas as pd, numpy as np, joblib
import autoencoder.autoencoder_triplet as autoencoder_triplet
import datasets.articles as articles
import helpers
from pathlib import Path
from autoencoder import utils


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
flags.DEFINE_string('input_format', 'binary', 'Input data format. ["binary", "tfidf"]')
flags.DEFINE_string('label', 'category_publish_name', 'Input data format. ["category_publish_name", "title_group"]')

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
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function. ["mean_squared", "cross_entropy", "cosine_proximity"]')
flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs, set to 0 will not train the model')
flags.DEFINE_float('batch_size', 0.1, 'Size of each mini-batch.')
flags.DEFINE_float('alpha', 1, 'hyper parameter for balancing similarity in loss function')

assert 0. <= FLAGS.min_df <= 1.
assert 0. <= FLAGS.max_df <= 1.
assert FLAGS.max_features >= 1
assert FLAGS.enc_act_func in ['sigmoid', 'tanh']
assert FLAGS.dec_act_func in ['sigmoid', 'tanh', 'none']
assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'decay', 'none']
assert 0. <= FLAGS.corr_frac <= 1.
assert FLAGS.loss_func in ['cross_entropy', 'mean_squared', 'cosine_proximity']
assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum']
assert FLAGS.verbose_step > 0
assert FLAGS.input_format in ['binary','tfidf']
assert FLAGS.label in ['category_publish_name','title_group']

if FLAGS.input_format == 'tfidf':
    assert FLAGS.loss_func in ['mean_squared', 'cosine_proximity']

if FLAGS.main_dir == '': FLAGS.main_dir = FLAGS.model_name


if __name__ == '__main__':
    print(__file__ + ': Start')

    ##################
    # init the model #
    ##################
    model = autoencoder_triplet.DenoisingAutoencoderTriplet(
        seed=FLAGS.seed, model_name=FLAGS.model_name, compress_factor=FLAGS.compress_factor,
        enc_act_func=FLAGS.enc_act_func, dec_act_func=FLAGS.dec_act_func, xavier_init=FLAGS.xavier_init,
        corr_type=FLAGS.corr_type, corr_frac=FLAGS.corr_frac,
        loss_func=FLAGS.loss_func, main_dir=FLAGS.main_dir, opt=FLAGS.opt,
        learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum,
        verbose=FLAGS.verbose, verbose_step=FLAGS.verbose_step, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
        alpha=FLAGS.alpha)

    # set row
    train_row = 1000
    validate_row = 10000

    ###########################
    # prepare or restore data #
    ###########################
    if FLAGS.restore_previous_data:
        # restore data
        article_contents = helpers.read_file(model.data_dir + 'article.snappy.parquet')
        article_contents.append(helpers.read_file(model.data_dir + 'article_validate.snappy.parquet'))
        X = helpers.read_file(model.data_dir + 'article_binary_count_vectorized.npz')
        X_pos = helpers.read_file(model.data_dir + 'article_binary_count_vectorized_pos.npz')
        X_neg = helpers.read_file(model.data_dir + 'article_binary_count_vectorized_neg.npz')
        X_validate = helpers.read_file(model.data_dir + 'article_binary_count_vectorized_validate.npz')
        X_validate_pos = helpers.read_file(model.data_dir + 'article_binary_count_vectorized_validate_pos.npz')
        X_validate_neg = helpers.read_file(model.data_dir + 'article_binary_count_vectorized_validate_neg.npz')
        X_label_category_publish_name = helpers.read_file(model.data_dir + 'article_label_category_publish_name.pkl', data_type='pandas_series')
        X_label_category_publish_name_validate = helpers.read_file(model.data_dir + 'article_label_category_publish_name_validate.pkl', data_type='pandas_series')
        X_label_title_group = helpers.read_file(model.data_dir + 'article_label_title_group.pkl', data_type='pandas_series')
        X_label_title_group_validate = helpers.read_file(model.data_dir + 'article_label_title_group_validate.pkl', data_type='pandas_series')
        X_tfidf = helpers.read_file(model.data_dir + 'article_tfidf_vectorized.npz')
        X_tfidf_pos = helpers.read_file(model.data_dir + 'article_tfidf_vectorized_pos.npz')
        X_tfidf_neg = helpers.read_file(model.data_dir + 'article_tfidf_vectorized_neg.npz')
        X_tfidf_validate = helpers.read_file(model.data_dir + 'article_tfidf_vectorized_validate.npz')
        X_tfidf_validate_pos = helpers.read_file(model.data_dir + 'article_tfidf_vectorized_validate_pos.npz')
        X_tfidf_validate_neg = helpers.read_file(model.data_dir + 'article_tfidf_vectorized_validate_neg.npz')
        count_vectorizer = joblib.load(model.data_dir + 'count_vectorizer.joblib')
        tfidf_transformer = joblib.load(model.data_dir + 'tfidf_transformer.joblib')

    else:
        article_contents = articles.read_articles(path='/Users/user/Documents/hk01/cache/s3/article_contents/latest.snappy.parquet')

        # get valid title group
        title_group_value_counts = article_contents.title_group.value_counts()
        title_group_indices = article_contents.title_group.isin(title_group_value_counts[(title_group_value_counts >= 2) & (title_group_value_counts <= 100)].index)
        title_group_indices = title_group_indices & ~title_group_indices.isin(['有片', '多圖', '今日天氣', '影片', '01影像', '熱評', '多圖有片', '多相', '片', '有圖慎入'])
        article_contents['label_title_group_valid'] = 0
        article_contents.loc[title_group_indices, 'label_title_group_valid'] = 1
        article_contents['label_title_group'] = pd.factorize(article_contents.title_group)[0]

        # get valid category
        def update_cate(cate_str):
            return cate_str.lstrip('即時')
        cate_value_counts = article_contents.category_publish_name.value_counts()
        cate_indices = article_contents.category_publish_name.isin(cate_value_counts[cate_value_counts > 100].index)
        cate_indices = cate_indices & ~cate_indices.isin(['突發', '熱話', '熱爆話題', '影片', '全部'])
        article_contents['label_category_publish_name_valid'] = 0
        article_contents.loc[cate_indices, 'label_category_publish_name_valid'] = 1
        article_contents['label_category_publish_name'] = pd.factorize(article_contents.category_publish_name.apply(update_cate))[0]

        article_contents = article_contents.loc[article_contents['label_' + FLAGS.label + '_valid'] == 1,]
        article_contents = articles.similar_articles(article_contents, id_colname='article_id',cate_colname='label_' + FLAGS.label, min_cate=2)

        count_vectorizer, X, X_pos, X_neg = articles.count_vectorize(
            article_contents[article_contents.valid_triplet_data == 1].main_content[0:train_row],
            article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_pos[0:train_row]],
            article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_neg[0:train_row]],
            min_df=FLAGS.min_df,
            max_df=FLAGS.max_df,
            max_features=FLAGS.max_features,
            binary=False
        )
        X_validate = count_vectorizer.transform(article_contents[article_contents.valid_triplet_data == 1].main_content[train_row:validate_row + train_row])
        X_validate_pos = count_vectorizer.transform(article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_pos[train_row:validate_row + train_row]])
        X_validate_neg = count_vectorizer.transform(article_contents.main_content.loc[article_contents[article_contents.valid_triplet_data == 1].article_id_neg[train_row:validate_row + train_row]])

        tfidf_transformer, X_tfidf = articles.tfidf_transform(X)
        X_tfidf_pos = tfidf_transformer.transform(X_pos)
        X_tfidf_neg = tfidf_transformer.transform(X_neg)
        X_tfidf_validate = tfidf_transformer.transform(X_validate)
        X_tfidf_validate_pos = tfidf_transformer.transform(X_validate_pos)
        X_tfidf_validate_neg = tfidf_transformer.transform(X_validate_neg)

        X_label_category_publish_name = article_contents.label_category_publish_name[0:train_row]
        X_label_category_publish_name_validate = article_contents.label_category_publish_name[train_row:validate_row + train_row]
        X_label_title_group = article_contents.label_title_group[0:train_row]
        X_label_title_group_validate = article_contents.label_title_group[train_row:validate_row + train_row]

        # Save training & validation data
        helpers.save_file(article_contents.iloc[0:train_row, ], model.data_dir + 'article.snappy.parquet')
        helpers.save_file(article_contents.iloc[train_row:validate_row + train_row, ], model.data_dir + 'article_validate.snappy.parquet')
        helpers.save_file(X_label_category_publish_name, model.data_dir + 'article_label_category_publish_name.pkl')
        helpers.save_file(X_label_category_publish_name_validate, model.data_dir + 'article_label_category_publish_name_validate.pkl')
        helpers.save_file(X_label_title_group, model.data_dir + 'article_label_title_group.pkl')
        helpers.save_file(X_label_title_group_validate, model.data_dir + 'article_label_title_group_validate.pkl')
        helpers.save_file(X, model.data_dir + 'article_count_vectorized.npz')
        helpers.save_file(X_validate, model.data_dir + 'article_count_vectorized_validate.npz')
        X.data = np.array([1] * len(X.data))
        X_pos.data = np.array([1] * len(X_pos.data))
        X_neg.data = np.array([1] * len(X_neg.data))
        X_validate.data = np.array([1] * len(X_validate.data))
        X_validate_pos.data = np.array([1] * len(X_validate_pos.data))
        X_validate_neg.data = np.array([1] * len(X_validate_neg.data))
        helpers.save_file(X, model.data_dir + 'article_binary_count_vectorized.npz')
        helpers.save_file(X_pos, model.data_dir + 'article_binary_count_vectorized_pos.npz')
        helpers.save_file(X_neg, model.data_dir + 'article_binary_count_vectorized_neg.npz')
        helpers.save_file(X_validate, model.data_dir + 'article_binary_count_vectorized_validate.npz')
        helpers.save_file(X_validate_pos, model.data_dir + 'article_binary_count_vectorized_validate_pos.npz')
        helpers.save_file(X_validate_neg, model.data_dir + 'article_binary_count_vectorized_validate_neg.npz')
        helpers.save_file(X_tfidf, model.data_dir + 'article_tfidf_vectorized.npz')
        helpers.save_file(X_tfidf_pos, model.data_dir + 'article_tfidf_vectorized_pos.npz')
        helpers.save_file(X_tfidf_neg, model.data_dir + 'article_tfidf_vectorized_neg.npz')
        helpers.save_file(X_tfidf_validate, model.data_dir + 'article_tfidf_vectorized_validate.npz')
        helpers.save_file(X_tfidf_validate_pos, model.data_dir + 'article_tfidf_vectorized_validate_pos.npz')
        helpers.save_file(X_tfidf_validate_neg, model.data_dir + 'article_tfidf_vectorized_validate_neg.npz')

        # Save in tsv format for visualization in tensorboard (http://projector.tensorflow.org/)
        helpers.save_file(X_tfidf, model.tsv_dir + 'article_tfidf_vectorized.tsv')
        helpers.save_file(X_tfidf_validate, model.tsv_dir + 'article_tfidf_vectorized.tsv')
        helpers.save_file(X, model.tsv_dir + 'article_binary_count_vectorized.tsv')
        helpers.save_file(X_validate, model.tsv_dir + 'article_binary_count_vectorized_validate.tsv')
        helpers.save_file(article_contents.iloc[0:train_row, ][['label_title_group', 'label_category_publish_name', 'title', 'title_group', 'category_publish_name']], model.tsv_dir + 'article_label.tsv')
        helpers.save_file(article_contents.iloc[train_row:validate_row + train_row, ][['label_title_group', 'label_category_publish_name', 'title', 'title_group', 'category_publish_name']], model.tsv_dir + 'article_label_validate.tsv')

        # Save vectorizer
        joblib.dump(count_vectorizer, model.data_dir + 'count_vectorizer.joblib')
        joblib.dump(tfidf_transformer, model.data_dir + 'tfidf_transformer.joblib')

    data_dict = {
        'binary': {
            'train': {'org': X,
                      'pos': X_pos,
                      'neg': X_neg},
            'validate': {'org': X_validate,
                         'pos': X_validate_pos,
                         'neg': X_validate_neg,},
        },
        'tfidf': {
            'train': {'org': X_tfidf,
                      'pos': X_tfidf_pos,
                      'neg': X_tfidf_neg,},
            'validate': {'org': X_tfidf_validate,
                         'pos': X_tfidf_validate_pos,
                         'neg': X_tfidf_validate_neg,},
        },
        'label_category_publish_name': {
            'train': X_label_category_publish_name,
            'validate': X_label_category_publish_name_validate,
        },
        'label_title_group': {
            'train': X_label_title_group,
            'validate': X_label_title_group_validate,
        },
    }

    trX = data_dict[FLAGS.input_format]['train']
    vlX = None
    if FLAGS.validation:
        vlX = data_dict[FLAGS.input_format]['validate']

    #################
    # Fit the model #
    #################
    print('fit')
    model.fit(train_set=trX, validation_set=vlX, restore_previous_model=FLAGS.restore_previous_model)
    # write parameter
    with open(model.parameter_file, 'a+') as text_file:
        print('train_row={}'.format(train_row), file=text_file)
        print('validate_row={}'.format(validate_row), file=text_file)
        print('input_format={}'.format(FLAGS.input_format), file=text_file)
        print('label={}'.format(FLAGS.label), file=text_file)
    print('fit done')

    # Encode the data and store it
    X_encoded = model.transform(utils.decay_noise(data_dict[FLAGS.input_format]['train']['org'], FLAGS.corr_frac), name='article_encoded', save=FLAGS.encode_full)
    X_encoded_validate = model.transform(utils.decay_noise(data_dict[FLAGS.input_format]['validate']['org'], FLAGS.corr_frac), name='article_encoded_validate', save=FLAGS.encode_full)
    helpers.save_file(X_encoded, model.tsv_dir + 'article_encoded.tsv')
    helpers.save_file(X_encoded_validate, model.tsv_dir + 'article_encoded_validate.tsv')

    #################################
    # Calculate pairwise similarity #
    #################################
    print('calculate similarity')
    article_binary_count_cosine_sim = helpers.pairwise_similarity(X, metric='cosine')
    del X
    article_binary_count_validate_cosine_sim = helpers.pairwise_similarity(X_validate, metric='cosine')
    del X_validate
    article_tfidf_cosine_sim = helpers.pairwise_similarity(X_tfidf, metric='linear kernel')  # This is same as cosine similarity as X_tfidf is l2 normalized (refer to sklearn's TFIDFTransformer for this)
    del X_tfidf
    article_tfidf_validate_cosine_sim = helpers.pairwise_similarity(X_tfidf_validate, metric='linear kernel')
    del X_tfidf_validate
    article_embedding_cosine_sim = helpers.pairwise_similarity(X_encoded, metric='cosine')
    del X_encoded
    article_embedding_validate_cosine_sim = helpers.pairwise_similarity(X_encoded_validate, metric='cosine')
    del X_encoded_validate
    print('calculate similarity done')

    #######################
    # Plot graph and save #
    #######################
    print('plot')
    for labels in ['label_category_publish_name', 'label_title_group']:
        suffix = '(Category)' if labels == 'label_category_publish_name' else '(Title Group)'
        helpers.visualize_pairwise_similarity(data_dict[labels]['train'], article_tfidf_cosine_sim, plot='boxplot',
                                              title='Cosine Similarity (TFIDF Vectorized) (Training Data)' + suffix,
                                              save_path=model.plot_dir + 'similarity_boxplot_tfidf' + suffix + '.png')
        helpers.visualize_pairwise_similarity(data_dict[labels]['validate'], article_tfidf_validate_cosine_sim, plot='boxplot',
                                              title='Cosine Similarity (TFIDF Vectorized) (Validation Data)' + suffix,
                                              save_path=model.plot_dir + 'similarity_boxplot_tfidf_validate' + suffix + '.png')

        helpers.visualize_pairwise_similarity(data_dict[labels]['train'], article_binary_count_cosine_sim, plot='boxplot',
                                              title='Cosine Similarity (Binary Count Vectorized) (Training Data)' + suffix,
                                              save_path=model.plot_dir + 'similarity_boxplot_binary_count' + suffix + '.png')
        helpers.visualize_pairwise_similarity(data_dict[labels]['validate'], article_binary_count_validate_cosine_sim, plot='boxplot',
                                              title='Cosine Similarity (Binary Count Vectorized) (Validation Data)' + suffix,
                                              save_path=model.plot_dir + 'similarity_boxplot_binary_count_validate' + suffix + '.png')

        helpers.visualize_pairwise_similarity(data_dict[labels]['train'], article_embedding_cosine_sim, plot='boxplot',
                                              title='Cosine Similarity (Encoded) (Training Data)' + suffix,
                                              save_path=model.plot_dir + 'similarity_boxplot_encoded' + suffix + '.png')
        helpers.visualize_pairwise_similarity(data_dict[labels]['validate'], article_embedding_validate_cosine_sim, plot='boxplot',
                                              title='Cosine Similarity (Encoded) (Validation Data)' + suffix,
                                              save_path=model.plot_dir + 'similarity_boxplot_encoded_validate' + suffix + '.png')
    print('plot done')

    ##########################
    # Print similar articles #
    ##########################
    article_binary_count_cosine_sim_argmax = np.nanargmax(article_binary_count_cosine_sim, 1)
    for i, v in enumerate(np.nanargmax(article_embedding_cosine_sim, 1)[0:5]):
        print(article_contents[['category_publish_name', 'title']].iloc[i])
        print('most similar article using count vectorizer')
        print(article_contents[['category_publish_name', 'title']].iloc[article_binary_count_cosine_sim_argmax[i]])
        print('most similar article using DAE')
        print(article_contents[['category_publish_name', 'title']].iloc[v])
        print('score: {}'.format(article_embedding_cosine_sim[i, v]))
        print()

    print(__file__ + ': End')

