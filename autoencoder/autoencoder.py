import tensorflow as tf
import numpy as np
import os, time
from pathlib import Path

from . import utils
from .triplet_loss_utils import batch_all_triplet_loss, batch_hard_triplet_loss, weighted_loss


# Define path
_script_path = Path(os.path.dirname(os.path.realpath(__file__)))


class DenoisingAutoencoder(object):

    """ Implementation of Denoising Autoencoders using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, algo_name='dae', model_name='dae', compress_factor=10, main_dir='dae/', enc_act_func='tanh',
                 dec_act_func='none', loss_func='mean_squared', num_epochs=10, batch_size=10,
                 xavier_init=1, opt='gradient_descent', learning_rate=0.01, momentum=0.5, corr_type='none',
                 corr_frac=0., verbose=True, verbose_step=5, seed=-1, alpha=1, triplet_strategy='batch_all'):
        """
        :param main_dir: main directory to put the models, data and summary directories
        :param compress_factor: number of hidden units = (input features divided by compress factor)
        :param enc_act_func: Activation function for the encoder. ['tanh', 'sigmoid']
        :param dec_act_func: Activation function for the decoder. ['tanh', 'sigmoid']
        :param loss_func: Loss function. ['mean_squared', 'cross_entropy']
        :param xavier_init: Value of the constant for xavier weights initialization
        :param opt: Which tensorflow optimizer to use. ['gradient_descent', 'momentum', 'ada_grad']
        :param learning_rate: Initial learning rate
        :param momentum: Momentum parameter
        :param corr_type: Type of input corruption. ["none", "masking", "salt_and_pepper", "decay]
        :param corr_frac: Fraction of the input to corrupt.
        :param verbose: Level of verbosity. False - silent, True - print accuracy.
        :param verbose_step: print accuracy every x training steps
        :param num_epochs: Number of epochs
        :param batch_size: Size of each mini-batch
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param alpha: balancing parameter between autoencoder loss and triplet loss
        :param triplet_strategy: triplet online mining strategy

        .. note: if triplet strategy is set to "none" then this is a typical denoising autoencoder with modification: H = f(Wx+b) - f(b)
        """

        self.algo_name = algo_name
        self.model_name = model_name
        self.compress_factor = compress_factor
        self.main_dir = main_dir
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.loss_func = loss_func
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.xavier_init = xavier_init
        self.opt = opt
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.corr_type = corr_type
        self.corr_frac = corr_frac
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.seed = seed
        self.alpha = alpha
        self.triplet_strategy = triplet_strategy

        assert type(self.verbose_step) == int
        assert self.verbose >= 0
        assert self.triplet_strategy in ['batch_all','batch_hard','none']

        if self.seed >= 0:
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)

        self.models_dir, self.data_dir, self.tf_summary_dir, self.tsv_dir, self.plot_dir = self._create_data_directories()
        self.model_path = self.models_dir + self.model_name

        self.sparse_input = None

        self.input_data = None
        self.input_data_corr = None

        self.W_ = None
        self.bh_ = None
        self.bv_ = None

        self.encode = None
        self.decode = None

        self.train_step = None
        self.cost = None

        self.tf_session = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_saver = None

        self.parameter_file = self.tf_summary_dir + 'parameter.txt'

    def _write_parameter_to_file(self, restore):
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

    def fit(self, train_set, validation_set=None, train_set_label=None, validation_set_label=None, restore_previous_model=False):
        """ Fit the model to the data.

        :param train_set: Training data.
        :param validation_set: optional, default None. Validation data.
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.

        :return: self
        """

        if self.triplet_strategy != 'none': assert train_set_label is not None
        if train_set_label is not None: assert train_set.shape[0] == len(train_set_label)
        if validation_set != None: assert validation_set.shape[0] == len(validation_set_label)

        n_features = train_set.shape[1]
        self.sparse_input = False if isinstance(train_set,np.ndarray) else True
        self.n_components = np.floor(n_features / self.compress_factor).astype(int)

        self._build_model(n_features)

        self._write_parameter_to_file(restore_previous_model)

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, validation_set, train_set_label=train_set_label, validation_set_label=validation_set_label)
            self.tf_summary_writer.close()
            self.tf_validation_summary_writer.close()
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

        self.tf_summary_writer = tf.summary.FileWriter(self.tf_summary_dir + 'train/', self.tf_session.graph)
        self.tf_validation_summary_writer = tf.summary.FileWriter(self.tf_summary_dir + 'validation/')

    def _train_model(self, train_set, validation_set, train_set_label, validation_set_label):

        """Train the model.

        :param train_set: training set
        :param validation_set: validation set. optional, default None
        :param train_set_label: training set label. only require when triplet strategy != "none"
        :param validation_set_label: validation set label. optional, default None

        :return: self
        """

        corruption_ratio = np.round(self.corr_frac * train_set.shape[1]).astype(np.int)

        for i in range(self.num_epochs):
            self.train_cost_batch = [], [], [] # corresponding to overall cost, autoencoder loss, triplet loss
            self.fraction_triplet_batch = [] # fraction of triplet (over all possible triplet combination) used for training
            self.num_triplet_batch = [] # number of triplet used for training
            train_start_time = time.time()

            self._run_train_step(train_set, train_set_label, corruption_ratio, i+1)

            self.train_time = time.time() - train_start_time

            if (i+1) % self.verbose_step == 0:
                self._run_validation_error_and_summaries(i+1, validation_set, validation_set_label)
        else:
            # run once when training is done
            if self.num_epochs!=0 and (i+1) % self.verbose_step != 0:
                self._run_validation_error_and_summaries(i+1, validation_set, validation_set_label)

    def _run_train_step(self, train_set, train_set_label, corruption_ratio, epoch):

        """ Run a training step. A training step is made by randomly corrupting the training set,
        randomly shuffling it,  divide it into batches and run the optimizer for each batch.

        :param train_set: training set
        :param train_set_label: training set label
        :param corruption_ratio: fraction of elements to corrupt

        :return: self
        """

        x_corrupted = self._corrupt_input(train_set, corruption_ratio)

        batches = [_ for _ in utils.gen_batches(train_set, x_corrupted, self.batch_size, data_label=train_set_label)]

        i = 1
        for batch in batches:
            if train_set_label is not None: x_batch, x_corr_batch, x_batch_label = batch
            else: x_batch, x_corr_batch = batch

            if self.sparse_input:
                tr_feed = {self.input_data: utils.get_sparse_ind_val_shape(x_batch), self.input_data_corr: utils.get_sparse_ind_val_shape(x_corr_batch), self.input_label: x_batch_label}
            else:
                tr_feed = {self.input_data: x_batch, self.input_data_corr: x_corr_batch, self.input_label: x_batch_label}

            if self.triplet_strategy != 'none':
                summary, step, train_autoencoder_loss, train_triplet_loss, train_cost, fraction_triplet, num_triplet = self.tf_session.run([self.tf_merged_summaries, self.train_step, self.autoencoder_loss, self.triplet_loss, self.cost, self.fraction_triplet, self.num_triplet], feed_dict=tr_feed)
                self.train_summary = summary
                self.train_cost_batch[0].append(train_cost)
                self.train_cost_batch[1].append(train_autoencoder_loss)
                self.train_cost_batch[2].append(train_triplet_loss)
                self.fraction_triplet_batch.append(fraction_triplet)
                self.num_triplet_batch.append(num_triplet)
            else:
                summary, step, train_cost = self.tf_session.run([self.tf_merged_summaries, self.train_step,self.cost], feed_dict=tr_feed)
                self.train_summary = summary
                self.train_cost_batch[0].append(train_cost)

            self.tf_summary_writer.add_summary(self.train_summary, (epoch - 1) * len(batches) + i)
            i += 1

    def _corrupt_input(self, data, v):

        """ Corrupt a fraction 'v' of 'data' according to the
        noise method of this autoencoder.
        :return: corrupted data
        """

        if self.corr_type == 'masking':
            x_corrupted = utils.masking_noise(data, self.corr_frac)

        elif self.corr_type == 'salt_and_pepper':
            x_corrupted = utils.salt_and_pepper_noise(data, v)

        elif self.corr_type == 'decay':
            x_corrupted = utils.decay_noise(data, self.corr_frac)

        elif self.corr_type == 'none':
            x_corrupted = data

        else:
            x_corrupted = None

        return x_corrupted

    def _run_validation_error_and_summaries(self, epoch, validation_set, validation_set_label):

        """ Run the summaries and error computation on the validation set.

        :param epoch: current epoch
        :param validation_set: validation data
        :param validation_set_label: validation data label

        :return: self
        """

        if self.verbose == 1:
            print('At step %d (%.2f seconds): ' % (epoch, self.train_time), end='')
            print('[Train Stat (average over past steps)] - ', end='')
            if self.triplet_strategy != 'none':
                print('Triplet: ', end='')
                print('Fraction=%.4f\t' % np.mean(self.fraction_triplet_batch), end='')
                print('Number=%.2f\t' % np.mean(self.num_triplet_batch), end='')
            print('Cost: ', end='')
            print('Overall=%.4f\t' % (np.mean(self.train_cost_batch[0])), end='')
            if self.triplet_strategy != 'none':
                print('Autoencoder=%.4f\t' % np.mean(self.train_cost_batch[1]), end='')
                print('Triplet=%.4f\t' % np.mean(self.train_cost_batch[2]), end='')

        if validation_set is None:
            print()
            return

        if self.sparse_input:
            _temp = utils.get_sparse_ind_val_shape(validation_set)
            vl_feed = {self.input_data: _temp, self.input_data_corr: _temp, self.input_label: validation_set_label}
        else:
            vl_feed = {self.input_data: validation_set, self.input_data_corr: validation_set, self.input_label: validation_set_label}

        if self.triplet_strategy != 'none':
            result = self.tf_session.run([self.tf_merged_summaries, self.cost, self.autoencoder_loss, self.triplet_loss], feed_dict=vl_feed)
        else:
            result = self.tf_session.run([self.tf_merged_summaries, self.cost], feed_dict=vl_feed)

        summary_str = result[0]
        self.tf_validation_summary_writer.add_summary(summary_str, epoch)

        if self.verbose:
            print("[Validation Stat (at this step)] - Cost: ")
            print('Overall=%.4f' % (result[1]), end='')
            if self.triplet_strategy != 'none':
                print('Autoencoder=%.4f\t' % (result[2]), end='')
                print('Triplet=%.4f\t' % (result[3]), end='')
            print()

    def _build_model(self, n_features):

        """ Creates the computational graph.

        :type n_features: int
        :param n_features: Number of features.

        :return: self
        """

        self.input_data, self.input_data_corr, self.input_label = self._create_placeholders()
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
        input_label = tf.placeholder('float', name='x-input-label')

        return input_data, input_data_corr, input_label

    def _create_variables(self, n_features):

        """ Create the TensorFlow variables for the model.

        :return: tuple(weights(shape(n_features, n_components)),
                       hidden bias(shape(n_components)),
                       visible bias(shape(n_features)))
        """

        W_ = tf.Variable(utils.xavier_init(n_features, self.n_components, self.xavier_init), name='enc-w')
        bh_ = tf.Variable(tf.zeros([self.n_components]), name='hidden-bias')
        bv_ = tf.Variable(tf.zeros([n_features]), name='visible-bias')

        return W_, bh_, bv_

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
                _enc_act_func = lambda x: x

            self.encode = _enc_act_func(_matmul(self.input_data_corr, self.W_) + self.bh_) - _enc_act_func(self.bh_)

            tf.summary.histogram('weights', self.W_)
            tf.summary.histogram('bias', self.bh_)
            tf.summary.histogram('embeddings', self.encode)

    def _create_decode_layer(self):

        """ Create the decoding layer of the network.
        :return: self
        """

        with tf.name_scope("Decode"):
            if self.dec_act_func == 'sigmoid':
                _dec_act_func = tf.nn.sigmoid

            elif self.dec_act_func == 'tanh':
                _dec_act_func = tf.nn.tanh

            else:
                _dec_act_func = lambda x: x

            self.decode = _dec_act_func(tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_)

            tf.summary.histogram('weights', tf.transpose(self.W_))
            tf.summary.histogram('bias', self.bv_)
            tf.summary.histogram('decodings', self.decode)

    def _create_cost_function_node(self):

        """ create the cost function node of the network.
        :return: self
        """

        with tf.name_scope("cost"):
            if self.triplet_strategy != 'none':
                if self.triplet_strategy == 'batch_all':
                    _triplet_loss = batch_all_triplet_loss
                elif self.triplet_strategy == 'batch_hard':
                    _triplet_loss = batch_hard_triplet_loss

                self.triplet_loss, data_weight, self.fraction_triplet, self.num_triplet = _triplet_loss(self.sparse_input, self.input_label, self.encode,)
                tf.summary.scalar('triplet_' + self.triplet_strategy, self.triplet_loss)

                self.autoencoder_loss = weighted_loss(self.sparse_input, self.input_data, self.decode, loss_func=self.loss_func, weight=data_weight)
                tf.summary.scalar('autoencoder_' + self.loss_func, self.autoencoder_loss)

                tf.summary.scalar('alpha', self.alpha)

                self.cost = self.autoencoder_loss + self.alpha * self.triplet_loss
                tf.summary.scalar('overall', self.cost)
            else:
                self.cost = weighted_loss(self.sparse_input, self.input_data, self.decode,loss_func=self.loss_func)
                tf.summary.scalar('autoencoder_' + self.loss_func, self.cost)

    def _create_train_step_node(self):

        """ create the training step node of the network.
        :return: self
        """

        with tf.name_scope("train"):
            if self.opt == 'gradient_descent':
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

                # Below are used for debug purpose only
                # [self.grad_W, self.grad_bh] = tf.gradients(self.cost, [self.W_, self.bh_])
                # self.new_W = self.W_.assign(self.W_ - self.learning_rate * self.grad_W)
                # self.new_bv = self.bv_.assign(self.bv_ - self.learning_rate * self.grad_bv)
                # self.new_bh = self.bh_.assign(self.bh_ - self.learning_rate * self.grad_bh)
                # self.train_step = [self.new_W, self.new_bh]

                # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                # self.grads_and_vars = self.optimizer.compute_gradients(self.cost, [self.W_, self.bv_, self.bh_])
                # self.train_step = self.optimizer.apply_gradients(self.grads_and_vars)

            elif self.opt == 'ada_grad':
                self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'momentum':
                self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)

            elif self.opt == 'adam':
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            else:
                self.train_step = None

        tf.summary.scalar('Learning rate', self.learning_rate)

    def transform(self, data, name='train', save=False):
        """ Transform data according to the model.

        :param data: Data to transform
        :param name: Identifier for the data that is being encoded
        :param save: If true, save data to disk

        :return: transformed data
        """

        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)

            if isinstance(data, np.ndarray):
                encoded_data = self.encode.eval({self.input_data_corr: data})
            else:
                # input_data_corr is a sparse tensor
                encoded_data = self.encode.eval({self.input_data_corr: utils.get_sparse_ind_val_shape(data)})

            weights = self.W_.eval()

            if save:
                np.save(self.data_dir + name, encoded_data)
                np.save(self.data_dir + 'weights', weights)

            return encoded_data

    def load_model(self, shape, model_path):
        """ Restore a previously trained model from disk.

        :param shape: tuple(n_features, n_components)
        :param model_path: path to the trained model

        :return: self, the trained model
        """
        self.n_components = shape[1]

        self._build_model(shape[0])

        init_op = tf.global_variables_initializer()

        self.tf_saver = tf.train.Saver()

        with tf.Session() as self.tf_session:

            self.tf_session.run(init_op)

            self.tf_saver.restore(self.tf_session, model_path)

    def get_model_parameters(self):
        """ Return the model parameters in the form of numpy arrays.

        :return: model parameters
        """
        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)

            return {
                'enc_w': self.W_.eval(),
                'enc_b': self.bh_.eval(),
                'dec_b': self.bv_.eval()
            }

    def _create_data_directories(self):

        """ Create the three directories for storing respectively the models,
        the data generated by training and the TensorFlow's summaries.

        :return: tuple of strings(models_dir, data_dir, summary_dir)
        """

        self.main_dir = (self.algo_name + '/' if self.algo_name[-1] != '/' else self.algo_name) + (self.main_dir + '/' if self.main_dir[-1] != '/' else self.main_dir)

        models_dir = 'results/' + self.main_dir + 'models/'
        data_dir = 'results/' + self.main_dir + 'data/'
        summary_dir = 'results/' + self.main_dir + 'logs/'
        tsv_dir = data_dir + 'tsv/'
        plot_dir = data_dir + 'plot/'

        for d in [models_dir, data_dir, summary_dir, tsv_dir, plot_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        return models_dir, data_dir, summary_dir, tsv_dir, plot_dir

    def get_weights_as_images(self, width, height, outdir='img/', max_images=10, model_path=None):
        """ Save the weights of this autoencoder as images, one image per hidden unit.
        Useful to visualize what the autoencoder has learned.

        :type width: int
        :param width: Width of the images

        :type height: int
        :param height: Height of the images

        :type outdir: string, default 'data/sdae/img'
        :param outdir: Output directory for the images. This path is appended to self.data_dir

        :type max_images: int, default 10
        :param max_images: Number of images to return.
        """
        assert max_images <= self.n_components

        outdir = self.data_dir + outdir

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        with tf.Session() as self.tf_session:

            if model_path is not None:
                self.tf_saver.restore(self.tf_session, model_path)
            else:
                self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)

            enc_weights = self.W_.eval()

            perm = np.random.permutation(self.n_components)[:max_images]

            for p in perm:

                enc_w = np.array([i[p] for i in enc_weights])
                image_path = outdir + self.model_name + '-enc_weights_{}.png'.format(p)
                utils.gen_image(enc_w, width, height, image_path)
