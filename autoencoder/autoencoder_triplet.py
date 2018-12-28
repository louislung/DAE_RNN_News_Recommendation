import tensorflow as tf
import numpy as np, os, time
from pathlib import Path

from . import utils
from .triplet_loss_utils import weighted_loss
from .autoencoder import DenoisingAutoencoder


# Define path
_script_path = Path(os.path.dirname(os.path.realpath(__file__)))


class DenoisingAutoencoderTriplet(DenoisingAutoencoder):

    """ Implementation of Denoising Autoencoders with Triplet using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, algo_name='dae_triplet', model_name='dae_triplet', compress_factor=10, main_dir='dae_triplet/', enc_act_func='tanh',
                 dec_act_func='none', loss_func='mean_squared', num_epochs=10, batch_size=10,
                 xavier_init=1, opt='gradient_descent', learning_rate=0.01, momentum=0.5, corr_type='none',
                 corr_frac=0., verbose=True, verbose_step=5, seed=-1, alpha=1):
        """
        :param alpha:
        :param refer to class DenoisingAutoencoder
        """

        super().__init__(algo_name=algo_name, model_name=model_name, compress_factor=compress_factor, main_dir=main_dir, enc_act_func=enc_act_func,
                         dec_act_func=dec_act_func, loss_func=loss_func, num_epochs=num_epochs, batch_size=batch_size,
                         xavier_init=xavier_init, opt=opt, learning_rate=learning_rate, momentum=momentum, corr_type=corr_type,
                         corr_frac=corr_frac, verbose=verbose, verbose_step=verbose_step, seed=seed, alpha=alpha, triplet_strategy='none')

        self.input_data_pos = None
        self.input_data_corr_pos = None

        self.input_data_neg = None
        self.input_data_corr_neg = None

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
        assert (train_set['pos'] != train_set['neg']).sum()

        if validation_set != None:
            assert type(validation_set['org']) == type(validation_set['pos'])
            assert type(validation_set['org']) == type(validation_set['neg'])
            assert validation_set['org'].shape == validation_set['pos'].shape
            assert validation_set['org'].shape == validation_set['neg'].shape
            assert (validation_set['pos'] != validation_set['neg']).sum()

        n_features = train_set['org'].shape[1]
        self.sparse_input = False if isinstance(train_set['org'],np.ndarray) else True
        self.n_components = np.floor(n_features / self.compress_factor).astype(int)

        self._build_model(n_features)

        self._write_parameter_to_file(restore_previous_model)

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, validation_set)
            self.tf_saver.save(self.tf_session, self.models_dir + self.model_name) #todo: should save to another model_name if model already exists?

    def _train_model(self, train_set, validation_set):

        """Train the model.

        :param train_set: dictionary of training set
        :param validation_set: validation set. optional, default None

        :return: self
        """

        corruption_ratio = np.round(self.corr_frac * train_set['org'].shape[1]).astype(np.int)

        for i in range(self.num_epochs):
            self.train_cost_batch = [], [], []
            train_start_time = time.time()

            self._run_train_step(train_set, corruption_ratio, i+1)

            self.train_time = time.time() - train_start_time

            if (i+1) % self.verbose_step == 0:
                self._run_validation_error_and_summaries(i+1, validation_set)
        else:
            # run once when training is done
            if self.num_epochs!=0 and (i+1) % self.verbose_step != 0:
                self._run_validation_error_and_summaries(i+1, validation_set)

    def _run_train_step(self, train_set, corruption_ratio, epoch):

        """ Run a training step. A training step is made by randomly corrupting the training set,
        randomly shuffling it,  divide it into batches and run the optimizer for each batch.

        :param train_set: dictionary of training set (org, pos, neg)
        :param corruption_ratio: fraction of elements to corrupt

        :return: self
        """

        x_corrupted = {}
        for key in train_set:
            x_corrupted[key] = self._corrupt_input(train_set[key], corruption_ratio)

        batches = [_ for _ in utils.gen_batches_triplet(train_set, x_corrupted, self.batch_size)]

        i = 1
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
            step, train_autoencoder_loss, train_triplet_loss, train_cost = self.tf_session.run([self.train_step,self.autoencoder_loss,self.triplet_loss,self.cost], feed_dict=tr_feed)

            self.train_cost_batch[0].append(train_cost)
            self.train_cost_batch[1].append(train_autoencoder_loss)
            self.train_cost_batch[2].append(train_triplet_loss)

            self.tf_summary_writer.add_summary(self.train_summary, (epoch - 1) * len(batches) + i)
            i += 1

    def _run_validation_error_and_summaries(self, epoch, validation_set):

        """ Run the summaries and error computation on the validation set.

        :param epoch: current epoch
        :param validation_set: validation data

        :return: self
        """

        if self.verbose == 1:
            print('At step %d (%.2f seconds): ' % (epoch, self.train_time), end='')
            print('[Train Stat (average over past steps)] - ', end='')
            print('Cost: ', end='')
            print('Overall=%.4f\t' % (np.mean(self.train_cost_batch[0])), end='')
            print('Autoencoder=%.4f\t' % np.mean(self.train_cost_batch[1]), end='')
            print('Triplet=%.4f\t' % np.mean(self.train_cost_batch[2]), end='')

        if validation_set is None:
            print()
            return

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

        result = self.tf_session.run([self.tf_merged_summaries, self.cost, self.autoencoder_loss, self.triplet_loss], feed_dict=vl_feed)

        summary_str = result[0]
        self.tf_validation_summary_writer.add_summary(summary_str, epoch)

        if self.verbose:
            print("[Validation Stat (at this step)] - Cost: ", end='')
            print('Overall=%.4f\t' % (result[1]), end='')
            print('Autoencoder=%.4f\t' % (result[2]), end='')
            print('Triplet=%.4f\t' % (result[3]), end='')
            print()

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

        with tf.name_scope("Encode"):
            if self.enc_act_func == 'sigmoid':
                _enc_act_func = tf.nn.sigmoid

            elif self.enc_act_func == 'tanh':
                _enc_act_func = tf.nn.tanh

            else:
                _enc_act_func = None

            self.encode = _enc_act_func(_matmul(self.input_data_corr, self.W_) + self.bh_) - _enc_act_func(self.bh_)
            self.encode_pos = _enc_act_func(_matmul(self.input_data_corr_pos, self.W_) + self.bh_) - _enc_act_func(self.bh_)
            self.encode_neg = _enc_act_func(_matmul(self.input_data_corr_neg, self.W_) + self.bh_) - _enc_act_func(self.bh_)

            tf.summary.histogram('weights', self.W_)
            tf.summary.histogram('bias', self.bh_)
            tf.summary.histogram('embeddings_anchor', self.encode)
            tf.summary.histogram('embeddings_pos', self.encode_pos)
            tf.summary.histogram('embeddings_neg', self.encode_neg)

    def _create_decode_layer(self):

        """ Create the decoding layer of the network.
        :return: self
        """

        with tf.name_scope("Decode"):
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

            tf.summary.histogram('weights', tf.transpose(self.W_))
            tf.summary.histogram('bias', self.bv_)
            tf.summary.histogram('decodings_anchor', self.decode)
            tf.summary.histogram('decodings_pos', self.decode_pos)
            tf.summary.histogram('decodings_neg', self.decode_neg)

    def _create_cost_function_node(self):

        """ create the cost function node of the network.
        :return: self
        """

        with tf.name_scope("cost"):
            self.autoencoder_loss = weighted_loss(self.sparse_input, self.input_data, self.decode, loss_func=self.loss_func) + \
                                    weighted_loss(self.sparse_input, self.input_data_pos, self.decode_pos, loss_func=self.loss_func) + \
                                    weighted_loss(self.sparse_input, self.input_data_neg, self.decode_neg, loss_func=self.loss_func)
            tf.summary.scalar('autoencoder_' + self.loss_func, self.autoencoder_loss)

            self.triplet_loss = tf.reduce_mean(-tf.log_sigmoid(tf.reduce_sum(
                (self.encode * self.encode_pos) -
                (self.encode * self.encode_neg)
                , 1)))
            tf.summary.scalar('triplet', self.triplet_loss)

            self.cost = self.autoencoder_loss + self.alpha * self.triplet_loss
            tf.summary.scalar("overall", self.cost)
