import tensorflow as tf
import numpy as np, os
from pathlib import Path

from . import utils
from .autoencoder import DenoisingAutoencoder


# Define path
_script_path = Path(os.path.dirname(os.path.realpath(__file__)))


class DenoisingAutoencoderTriplet(DenoisingAutoencoder):

    """ Implementation of Denoising Autoencoders with Triplet using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, model_name='dae_triplet', compress_factor=10, main_dir='dae_triplet/', enc_act_func='tanh',
                 dec_act_func='none', loss_func='mean_squared', num_epochs=10, batch_size=10,
                 xavier_init=1, opt='gradient_descent', learning_rate=0.01, momentum=0.5, corr_type='none',
                 corr_frac=0., verbose=True, seed=-1, alpha=1):
        """
        :param alpha:
        :param refer to class DenoisingAutoencoder
        """

        self.alpha = alpha

        self.input_data_pos = None
        self.input_data_corr_pos = None

        self.input_data_neg = None
        self.input_data_corr_neg = None

        super().__init__(model_name, compress_factor, main_dir, enc_act_func,
                         dec_act_func, loss_func, num_epochs, batch_size,
                         xavier_init, opt, learning_rate, momentum, corr_type,
                         corr_frac, verbose, seed)

    def fit(self, train_set, validation_set=None, restore_previous_model=False):
        """ Fit the model to the data.

        :param train_set: Dictionary of training data (org, pos, neg).
        :param validation_set: optional, default None. Dictionary of validation data (org, pos, neg).
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.

        :return: self
        """

        assert type(train_set['org']) == type(train_set['pos'])
        assert type(train_set['org']) == type(train_set['neg'])
        assert train_set['org'].shape == train_set['pos'].shape
        assert train_set['org'].shape == train_set['neg'].shape

        if validation_set != None:
            assert type(validation_set['org']) == type(validation_set['pos'])
            assert type(validation_set['org']) == type(validation_set['neg'])
            assert validation_set['org'].shape == validation_set['pos'].shape
            assert validation_set['org'].shape == validation_set['neg'].shape

        n_features = train_set['org'].shape[1]
        self.sparse_input = False if isinstance(train_set['org'],np.ndarray) else True
        self.n_components = np.floor(n_features / self.compress_factor).astype(int)

        self._build_model(n_features)

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, validation_set)
            self.tf_saver.save(self.tf_session, self.models_dir + self.model_name) #todo: should save to another model_name if model already exists?

    def _initialize_tf_utilities_and_ops(self, restore_previous_model):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model is true.
        """

        self.tf_merged_summaries = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

        self.tf_summary_writer = tf.summary.FileWriter(self.tf_summary_dir, self.tf_session.graph)

    def _train_model(self, train_set, validation_set):

        """Train the model.

        :param train_set: dictionary of training set
        :param validation_set: validation set. optional, default None

        :return: self
        """

        corruption_ratio = np.round(self.corr_frac * train_set['org'].shape[1]).astype(np.int)

        for i in range(self.num_epochs):
            self._run_train_step(train_set, corruption_ratio)

            if i % 5 == 0:
                if validation_set is not None:
                    self._run_validation_error_and_summaries(i, validation_set)

    def _run_train_step(self, train_set, corruption_ratio):

        """ Run a training step. A training step is made by randomly corrupting the training set,
        randomly shuffling it,  divide it into batches and run the optimizer for each batch.

        :param train_set: dictionary of training set (org, pos, neg)
        :param corruption_ratio: fraction of elements to corrupt

        :return: self
        """

        x_corrupted = {}
        for key in train_set:
            x_corrupted[key] = self._corrupt_input(train_set[key], corruption_ratio)

        batches = [_ for _ in utils.gen_batches_triplet(train_set, x_corrupted, self.batch_size, self.sparse_input)]

        for batch in batches:
            x_batch, x_corr_batch = batch
            if self.sparse_input:
                tr_feed = {self.input_data: utils.get_sparse_ind_val_shape(x_batch[0]),
                           self.input_data_pos: utils.get_sparse_ind_val_shape(x_batch[1]),
                           self.input_data_neg: utils.get_sparse_ind_val_shape(x_batch[2]),
                           self.input_data_corr: utils.get_sparse_ind_val_shape(x_corr_batch[0]),
                           self.input_data_corr_pos: utils.get_sparse_ind_val_shape(x_corr_batch[1]),
                           self.input_data_corr_neg: utils.get_sparse_ind_val_shape(x_corr_batch[2])}
            else:
                tr_feed = {self.input_data: x_batch[0],
                           self.input_data_pos: x_batch[1],
                           self.input_data_neg: x_batch[2],
                           self.input_data_corr: x_corr_batch[0],
                           self.input_data_corr_pos: x_corr_batch[1],
                           self.input_data_corr_neg: x_corr_batch[2]}
            self.tf_session.run(self.train_step, feed_dict=tr_feed)

    def _run_validation_error_and_summaries(self, epoch, validation_set):

        """ Run the summaries and error computation on the validation set.

        :param epoch: current epoch
        :param validation_set: validation data

        :return: self
        """

        if self.sparse_input:
            _temp = utils.get_sparse_ind_val_shape(validation_set['org'])
            _temp_pos = utils.get_sparse_ind_val_shape(validation_set['pos'])
            _temp_neg = utils.get_sparse_ind_val_shape(validation_set['neg'])
            vl_feed = {self.input_data: _temp,
                       self.input_data_pos: _temp_pos,
                       self.input_data_neg: _temp_neg,
                       self.input_data_corr: _temp,
                       self.input_data_corr_pos: _temp_pos,
                       self.input_data_corr_neg: _temp_neg}
        else:
            vl_feed = {self.input_data: validation_set['org'],
                       self.input_data_pos: validation_set['pos'],
                       self.input_data_neg: validation_set['neg'],
                       self.input_data_corr: validation_set['org'],
                       self.input_data_corr_pos: validation_set['pos'],
                       self.input_data_corr_neg: validation_set['neg']}

        result = self.tf_session.run([self.tf_merged_summaries, self.cost], feed_dict=vl_feed)
        summary_str = result[0]
        err = result[1]

        self.tf_summary_writer.add_summary(summary_str, epoch)

        if self.verbose == 1:
            print("Validation cost at step %s: %s" % (epoch, err))

    def _build_model(self, n_features):

        """ Creates the computational graph.

        :type n_features: int
        :param n_features: Number of features.

        :return: self
        """

        self.input_data, self.input_data_corr, self.input_data_pos, self.input_data_corr_pos, self.input_data_neg, self.input_data_corr_neg = self._create_placeholders()
        self.W_, self.bh_, self.bv_ = self._create_variables(n_features)

        self._create_encode_layer()
        self._create_decode_layer()

        self._create_cost_function_node()
        self._create_train_step_node()

    def _create_placeholders(self):

        """ Create the TensorFlow (sparse) placeholders for the model.

        :return: tuple(input_data(shape(None, n_features)),
                       input_data_corr(shape(None, n_features)))
        """

        _placeholder = tf.sparse.placeholder if self.sparse_input else tf.placeholder
        input_data = _placeholder('float', name='x-input')
        input_data_corr = _placeholder('float', name='x-corr-input')
        input_data_pos = _placeholder('float', name='x-input-positive')
        input_data_corr_pos = _placeholder('float', name='x-corr-input-positive')
        input_data_neg = _placeholder('float', name='x-input-negative')
        input_data_corr_neg = _placeholder('float', name='x-corr-input-negative')

        return input_data, input_data_corr, input_data_pos, input_data_corr_pos, input_data_neg, input_data_corr_neg

    def _create_encode_layer(self):

        """ Create the encoding layer of the network.
        :return: self
        """

        _matmul = tf.sparse.matmul if self.sparse_input else tf.matmul

        with tf.name_scope("W_x_bh"):
            if self.enc_act_func == 'sigmoid':
                _enc_act_func = tf.nn.sigmoid

            elif self.enc_act_func == 'tanh':
                _enc_act_func = tf.nn.tanh

            else:
                _enc_act_func = None

            self.encode = _enc_act_func(_matmul(self.input_data_corr, self.W_) + self.bh_) - _enc_act_func(self.bh_)
            self.encode_pos = _enc_act_func(_matmul(self.input_data_corr_pos, self.W_) + self.bh_) - _enc_act_func(self.bh_)
            self.encode_neg = _enc_act_func(_matmul(self.input_data_corr_neg, self.W_) + self.bh_) - _enc_act_func(self.bh_)

    def _create_decode_layer(self):

        """ Create the decoding layer of the network.
        :return: self
        """

        with tf.name_scope("Wg_y_bv"):
            if self.dec_act_func == 'sigmoid':
                _dec_act_func = tf.nn.sigmoid
                self.decode = tf.nn.sigmoid

            elif self.dec_act_func == 'tanh':
                _dec_act_func = tf.nn.tanh

            elif self.dec_act_func == 'none':
                _dec_act_func = lambda x: x

            else:
                _dec_act_func = None

            self.decode = _dec_act_func(tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_)
            self.decode_pos = _dec_act_func(tf.matmul(self.encode_pos, tf.transpose(self.W_)) + self.bv_)
            self.decode_neg = _dec_act_func(tf.matmul(self.encode_neg, tf.transpose(self.W_)) + self.bv_)

    def _create_cost_function_node(self):

        """ create the cost function node of the network.
        :return: self
        """

        _reduce_sum = tf.sparse.reduce_sum if self.sparse_input else tf.reduce_sum

        with tf.name_scope("cost"):
            if self.loss_func == 'cross_entropy':
                self.cost = - _reduce_sum(self.input_data.__mul__(tf.log(self.decode)))
                self.cost += - _reduce_sum(self.input_data_pos.__mul__(tf.log(self.decode_pos)))
                self.cost += - _reduce_sum(self.input_data_neg.__mul__(tf.log(self.decode_neg)))
                self.cost += self.alpha * tf.reduce_sum(tf.log1p(tf.exp(
                    tf.matmul(self.encode, tf.transpose(self.encode_neg)) -
                    tf.matmul(self.encode, tf.transpose(self.encode_pos))
                )))
                _ = tf.summary.scalar("cross_entropy", self.cost)

            elif self.loss_func == 'mean_squared':
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - self.decode)))
                _ = tf.summary.scalar("mean_squared", self.cost)

            else:
                self.cost = None

    def _create_train_step_node(self):

        """ create the training step node of the network.
        :return: self
        """

        with tf.name_scope("train"):
            if self.opt == 'gradient_descent':
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'ada_grad':
                self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'momentum':
                self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)

            else:
                self.train_step = None