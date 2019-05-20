
from . import data
import codecs
import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

# word level special token
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

class Model(object):

    def __init__(self, input_file, target_file, vocab_file, mode, num_units, batch_size):
        self.input_file = input_file
        self.target_file = target_file
        self.vocab_file = vocab_file

        self.mode = mode
        self.num_units = num_units
        self.batch_size = batch_size
        self.vocab_table = lookup_ops.index_table_from_file(vocab_file, default_value=UNK_ID)
        #self.vocabs, vocab_size = load_vocab(vocab_file)

        if self.mode == 'train':
            self._init_train()

        self._init_model
        self._init_infer
        return

    def load_vocab(vocab_file):
        vocab = []
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            for word in f:
                vocab.append(word.strip())
        return vocab, len(vocab)

    def _init_iterator(self):

        sos_id = tf.cast(src_vocab_table.lookup(tf.constant(SOS)), tf.int32)
        eos_id = tf.cast(src_vocab_table.lookup(tf.constant(EOS)), tf.int32)


        src_dataset = tf.data.TextLineDataset(self.src_file)
        target_dataset = tf.data.TextLineDataset(self.target_file)
        src_target_dataset = tf.data.Dataset.zip((src_dataset, target_dataset))

        output_buffer_size = self.batch_size * 1000
        num_parallel_calls = 4
        src_target_dataset = src_target_dataset.shuffle(output_buffer_size, reshuffle_each_iteration=True)

        src_target_dataset = src_target_dataset.map(lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
                num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        src_target_dataset = src_target_dataset.map(lambda src, tgt: (tf.cast(self.vocab_table.lookup(src), tf.int32),
            tf.cast(self.vocab_table.lookup(tgt), tf.int32)), num_parallel_calls=num_parallel_calls)

        src_target_dataset = src_target_dataset.prefetch(output_buffer_size)

        src_target_dataset = src_target_dataset.map(lambda src, tgt: (src,
            tf.concat(([sos_id], tgt), 0),
            tf.concat((tgt, [eos_id]), 0)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

        src_target_dataset = src_target_dataset.map(lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
            num_parallel_calls=num_parallel_calls)

        src_target_dataset = src_target_dataset.prefetch(output_buffer_size)


        batched_dataset = input_target_dataset.padded_batch(
                self.batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # tgt_input
                    tf.TensorShape([None]),  # tgt_output
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([])),  # tgt_len
                # Pad the source and target sequences with eos tokens.
                # (Though notice we don't generally need to do this since
                # later on we will be masking out calculations past the true sequence.
                padding_values=(
                    src_eos_id,  # src
                    tgt_eos_id,  # tgt_input
                    tgt_eos_id,  # tgt_output
                    0,  # src_len -- unused
                    0))  # tgt_len -- unused
        batched_iterator = batched_dataset.make_initializable_iterator()
        return batched_iterator


    def _init_train(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/cpu:0'):
                embedding = tf.get_variable("embedding", [self.vocab_size, self.num_units])
            encoder_emb_inp = tf.nn.embedding_lookup(embedding, in_seq, name='embed_input')

        forward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        backward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                forward_cell, backward_cell, encoder_emb_inp,
                sequence_length=source_sequence_length, time_major=True)
        encoder_outputs = tf.concat(bi_outputs, -1)

        attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, attention_states,
            memory_sequence_length=source_sequence_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=num_units)

        helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, decoder_lengths, time_major=True)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, encoder_state,
                output_layer=projection_layer)
        # Dynamic decoding
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
        logits = outputs.rnn_output

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=decoder_outputs, logits=logits)
        train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        # Optimization
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_step = optimizer.apply_gradients(
            zip(clipped_gradients, params))

    def _init_eval(self):
        return

    def _init_infer(self):
        # Helper
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder,
            tf.fill([batch_size], tgt_sos_id), tgt_eos_id)

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,
            output_layer=projection_layer)

        # Dynamic decoding
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        translations = outputs.sample_id


    def train():
        train_graph = tf.Graph()
        eval_graph = tf.Graph()
        infer_graph = tf.Graph()

        with train_graph.as_default():
            train_iterator = ...
            train_model = BuildTrainModel(train_iterator)
            initializer = tf.global_variables_initializer()

        with eval_graph.as_default():
            eval_iterator = ...
            eval_model = BuildEvalModel(eval_iterator)

        with infer_graph.as_default():
            infer_iterator, infer_inputs = ...
            infer_model = BuildInferenceModel(infer_iterator)

        checkpoints_path = "/tmp/model/checkpoints"

        train_sess = tf.Session(graph=train_graph)
        eval_sess = tf.Session(graph=eval_graph)
        infer_sess = tf.Session(graph=infer_graph)

        train_sess.run(initializer)
        train_sess.run(train_iterator.initializer)

        for i in itertools.count():
            train_model.train(train_sess)

            if i % EVAL_STEPS == 0:
                checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=i)
                eval_model.saver.restore(eval_sess, checkpoint_path)
                eval_sess.run(eval_iterator.initializer)
                while data_to_eval:
                    eval_model.eval(eval_sess)

            if i % INFER_STEPS == 0:
                checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=i)
                infer_model.saver.restore(infer_sess, checkpoint_path)
                infer_sess.run(infer_iterator.initializer, feed_dict={infer_inputs: infer_input_data})
                while data_to_infer:
                infer_model.infer(infer_sess)
