import tensorflow as tf
import numpy as np
from qa_data import PAD_ID
from tensorflow.python.ops import variable_scope as vs
import time
import logging


class Seq2Seq:

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.setup_placeholders()
        self.setup_embeddings()
        self.encode()
        self.decode()
        self.init_op = tf.global_variables_initializer()

    def initialize(self, session):
        session.run(self.init_op)

    def setup_placeholders(self):
        # self.q_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="q_place") #batch by seq (None, None)
        # self.p_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="p_place")
        # self.q_mask_placeholder = tf.placeholder(tf.int32, shape=[None], name="q_mask") #batch (None)
        # self.p_mask_placeholder = tf.placeholder(tf.int32, shape=[None], name="p_mask")

        self.encs_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="encs_place")
        self.decs_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="decs_place")
        self.encs_len_placeholder = tf.placeholder(tf.int32, shape=[None], name="encs_mask")
        self.decs_len_placeholder = tf.placeholder(tf.int32, shape=[None], name="decs_mask")

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embedding_files = np.load("dialog_corpus/movie/glove.trimmed.100.npz")
            self.pretrained_embeddings = tf.constant(embedding_files["glove"], dtype=tf.float32)
            self.encs = tf.nn.embedding_lookup(self.pretrained_embeddings, self.encs_placeholder)
            self.decs = tf.nn.embedding_lookup(self.pretrained_embeddings, self.decs_placeholder)

    def encode(self):
        hidden_size = 42
        self.test_size = tf.shape(self.encs)
        cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        outputs, final_state = tf.nn.dynamic_rnn(cell, self.encs, sequence_length=self.encs_len_placeholder, dtype=tf.float32)
        self.encode_output = outputs
        self.encode_thought = final_state

    def decode(self):
        hidden_size = 42
        self.test_1 = tf.shape(self.encode_output)
        self.test_2 = tf.shape(self.encode_thought)

    def make_batch(self, dataset, iteration):
        batch_size = self.FLAGS.batch_size
        train_enc, train_dec, val_enc, val_dec = dataset
        start_index = iteration*batch_size

        #make padded enc batch
        encs = train_enc[start_index:start_index+batch_size]
        encs_len = np.array([len(q) for q in encs])
        encs_max_len = np.max(encs_len)
        encs_batch = np.array([q + [PAD_ID]*(encs_max_len - len(q)) for q in encs])

        #make padded dec batch
        decs = train_dec[start_index:start_index+batch_size]
        decs_len= np.array([len(p) for p in decs])
        decs_max_len = np.max(decs_len)
        decs_batch = np.array([p + [PAD_ID]*(decs_max_len - len(p)) for p in decs])

        return encs_batch, encs_len, decs_batch, decs_len

    # def build_model(self):
    #     self.test =
    def optimize(self, session, data):
        encs_batch, encs_len, decs_batch, decs_len = data
        feed_dict = {}
        feed_dict[self.encs_placeholder] = encs_batch
        feed_dict[self.encs_len_placeholder] = encs_len
        feed_dict[self.decs_placeholder] = decs_batch
        feed_dict[self.decs_len_placeholder] = decs_len

        output_feed = [self.test_1, self.test_2]
        return session.run(output_feed, feed_dict)

    def train(self, session, dataset, train_dir):

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))


        #run main training loop: (only 1 epoch for now)
        train_enc, train_dec, val_enc, val_dec = dataset
        max_iters = np.ceil(len(train_enc)/float(self.FLAGS.batch_size))
        print("Max iterations: " + str(max_iters))
        for epoch in range(1):
            #temp hack to only train on some small subset:
            max_iters = 1
            for iteration in range(int(max_iters)):
                print("Current iteration: " + str(iteration))
                encs_batch, encs_len, decs_batch, decs_len = self.make_batch(dataset, iteration)
                # lr = tf.train.exponential_decay(self.FLAGS.learning_rate, iteration, 100, 0.96) #iteration here should be global when multiple epochs
                #retrieve useful info from training - see optimize() function to set what we're tracking
                output_shape, final_state_shape = self.optimize(session, (encs_batch, encs_len, decs_batch, decs_len))
                print(output_shape)
                print(final_state_shape)
                # print("Current Loss: " + str(loss))
                # print(grad_norm)
