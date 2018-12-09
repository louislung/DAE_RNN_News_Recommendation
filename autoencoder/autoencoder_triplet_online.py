import tensorflow as tf
import numpy as np, os, time
from pathlib import Path

from . import utils
from .autoencoder import DenoisingAutoencoder
from .triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss

# Define path
_script_path = Path(os.path.dirname(os.path.realpath(__file__)))


class DenoisingAutoencoderTripletOnline(DenoisingAutoencoder):

    """ Implementation of Denoising Autoencoders with Triplet Online Learning using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, algo_name='dae_triplet_online', model_name='dae_triplet_online', compress_factor=10, main_dir='dae_triplet_online/', enc_act_func='tanh',
                 dec_act_func='none', loss_func='mean_squared', num_epochs=10, batch_size=10,
                 xavier_init=1, opt='gradient_descent', learning_rate=0.01, momentum=0.5, corr_type='none',
                 corr_frac=0., verbose=True, verbose_step=5, seed=-1, alpha=1, triplet_strategy='batch_all'):
        """
        :param alpha:
        :param refer to class DenoisingAutoencoder
        """

        self.alpha = alpha

        self.train_cost = ([], [], [])
        self.fraction = []
        self.num = []

        self.triplet_strategy = triplet_strategy

        super().__init__(algo_name, model_name, compress_factor, main_dir, enc_act_func,
                         dec_act_func, loss_func, num_epochs, batch_size,
                         xavier_init, opt, learning_rate, momentum, corr_type,
                         corr_frac, verbose, verbose_step, seed)

    def _write_parameter_to_file(self, restore):
        self.parameter_file = self.tf_summary_dir + 'parameter.txt'
        mode = 'a+' if restore else 'w'
        with open(self.parameter_file, mode) as text_file:
            print('---------------------------------------', file=text_file)
            print('algo_name={}'.format(self.algo_name), file=text_file)
            print('model_name={}'.format(self.model_name), file=text_file)
            print('compress_factor={}'.format(self.compress_factor), file=text_file)
            print('main_dir={}'.format(self.main_dir), file=text_file)
            print('enc_act_func={}'.format(self.enc_act_func), file=text_file)
            print('dec_act_func={}'.format(self.dec_act_func), file=text_file)
            print('loss_func={}'.format(self.loss_func), file=text_file)
            print('num_epochs={}'.format(self.num_epochs), file=text_file)
            print('batch_size={}'.format(self.batch_size), file=text_file)
            print('xavier_init={}'.format(self.xavier_init), file=text_file)
            print('opt={}'.format(self.opt), file=text_file)
            print('learning_rate={}'.format(self.learning_rate), file=text_file)
            print('momentum={}'.format(self.momentum), file=text_file)
            print('corr_type={}'.format(self.corr_type), file=text_file)
            print('corr_frac={}'.format(self.corr_frac), file=text_file)
            print('verbose={}'.format(self.verbose), file=text_file)
            print('verbose_step={}'.format(self.verbose_step), file=text_file)
            print('seed={}'.format(self.seed), file=text_file)
            print('alpha={}'.format(self.alpha), file=text_file)
            print('triplet_strategy={}'.format(self.triplet_strategy), file=text_file)

    def fit(self, train_set, train_set_label, validation_set=None, validation_set_label=None, restore_previous_model=False):
        """ Fit the model to the data.
        #todo: update desc
        :param train_set: Dictionary of training data (org, pos, neg).
        :param validation_set: optional, default None. Dictionary of validation data (org, pos, neg).
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.

        :return: self
        """

        assert train_set.shape[0] == len(train_set_label)

        if validation_set != None:
            assert validation_set.shape[0] == len(validation_set_label)

        n_features = train_set.shape[1]
        self.sparse_input = False if isinstance(train_set,np.ndarray) else True
        self.n_components = np.floor(n_features / self.compress_factor).astype(int)

        self.input_label = tf.placeholder('float', name='x-input-label')
        self._build_model(n_features)

        self._write_parameter_to_file(restore_previous_model)

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, train_set_label, validation_set, validation_set_label)
            self.tf_saver.save(self.tf_session, self.models_dir + self.model_name) #todo: should save to another model_name if model already exists?

    def _train_model(self, train_set, train_set_label, validation_set, validation_set_label):

        """Train the model.
        #todo: update desc
        :param train_set: training set
        :param validation_set: validation set. optional, default None

        :return: self
        """

        corruption_ratio = np.round(self.corr_frac * train_set.shape[1]).astype(np.int)

        for i in range(self.num_epochs):
            self.train_cost = ([], [], [])
            self.fraction = []
            self.num = []
            train_start_time = time.time()
            self._run_train_step(train_set, train_set_label, corruption_ratio)
            self.train_time = time.time() - train_start_time

            if (i+1) % self.verbose_step == 0:
                self._run_validation_error_and_summaries(i+1, validation_set, validation_set_label)

        else:
            if self.num_epochs!=0 and (i+1) % self.verbose_step != 0:
                self._run_validation_error_and_summaries(self.num_epochs, validation_set, validation_set_label)

    def _run_train_step(self, train_set, train_set_label, corruption_ratio):

        """ Run a training step. A training step is made by randomly corrupting the training set,
        randomly shuffling it,  divide it into batches and run the optimizer for each batch.

        :param train_set: training set
        :param corruption_ratio: fraction of elements to corrupt

        :return: self
        """

        x_corrupted = self._corrupt_input(train_set, corruption_ratio)

        batches = [_ for _ in utils.gen_batches(train_set, x_corrupted, self.batch_size, self.sparse_input, train_set_label)]

        for batch in batches:
            x_batch, x_corr_batch, x_batch_label = batch
            if self.sparse_input:
                tr_feed = {self.input_data: utils.get_sparse_ind_val_shape(x_batch), self.input_data_corr: utils.get_sparse_ind_val_shape(x_corr_batch), self.input_label: x_batch_label}
            else:
                tr_feed = {self.input_data: x_batch, self.input_data_corr: x_corr_batch, self.input_label: x_batch_label}
            step, train_autoencoder_loss, train_triplet_loss, train_cost, fraction_positive_triplet, num_positive_triplet = self.tf_session.run([self.train_step, self.autoencoder_loss, self.triplet_loss, self.cost, self.fraction_positive_triplet, self.num_positive_triplet], feed_dict=tr_feed)

            self.train_cost[0].append(train_autoencoder_loss)
            self.train_cost[1].append(train_triplet_loss)
            self.train_cost[2].append(train_cost)
            self.fraction.append(fraction_positive_triplet)
            self.num.append(num_positive_triplet)

    def _run_validation_error_and_summaries(self, epoch, validation_set, validation_set_label):

        """ Run the summaries and error computation on the validation set.

        :param epoch: current epoch
        :param validation_set: validation data

        :return: self
        """

        if self.verbose == 1:
            print('At step %d (%.2f seconds): ' % (epoch, self.train_time), end='')
            print('Positive Triplet: ', end='')
            if (self.fraction.count(None) != len(self.fraction)): print('Fraction=%.4f\t' % np.mean(self.fraction), end='')
            if (self.num.count(None) != len(self.num)): print('Number=%d\t' % np.mean(self.num), end='')
            print('Training cost: ', end='')
            print('Autoencoder=%.4f\t' % np.mean(self.train_cost[0]), end='')
            print('Triplet=%.4f\t' % np.mean(self.train_cost[1]), end='')
            print('Overall=%.4f' % (np.mean(self.train_cost[2])), end='')
            print()

        if validation_set is None: return

        if self.sparse_input:
            _temp = utils.get_sparse_ind_val_shape(validation_set)
            vl_feed = {self.input_data: _temp, self.input_data_corr: _temp, self.input_label: validation_set_label}
        else:
            vl_feed = {self.input_data: validation_set, self.input_data_corr: validation_set, self.input_label: validation_set_label}

        result = self.tf_session.run([self.tf_merged_summaries, self.autoencoder_loss, self.triplet_loss, self.cost], feed_dict=vl_feed)
        summary_str = result[0]

        self.tf_validation_summary_writer.add_summary(summary_str, epoch)

        if self.verbose == 1:
            print("\tValidation cost: Autoencoder=%.4f\tTriplet=%.4f\tOverall=%.4f" % (result[1], result[2], result[3]), end='')
            print()

    def _create_cost_function_node(self):

        """ create the cost function node of the network.
        :return: self
        """

        _reduce_sum = tf.sparse.reduce_sum if self.sparse_input else tf.reduce_sum

        with tf.name_scope("cost"):
            if self.loss_func == 'cross_entropy':
                if self.triplet_strategy == 'batch_all':
                    _batch_loss = batch_all_triplet_loss
                elif self.triplet_strategy == 'batch_hard':
                    _batch_loss = batch_hard_triplet_loss

                self.autoencoder_loss, self.triplet_loss, self.fraction_positive_triplet, self.num_positive_triplet = _batch_loss(self.sparse_input, self.input_label, self.input_data, self.encode, self.decode)
                self.cost = self.autoencoder_loss + self.alpha * self.triplet_loss
                _ = tf.summary.scalar("cross_entropy", self.cost)

            #elif self.loss_func == 'mean_squared':
            #    self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - self.decode)))
            #    _ = tf.summary.scalar("mean_squared", self.cost)

            else:
                self.cost = None