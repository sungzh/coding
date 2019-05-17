#coding=utf-8

import json

from tensorflow.python.ops import lookup_ops

# word level special token
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

class data(object):

    def __init__(self, input_files, target_files, vocab_files, batch_size):
        self.ts1 = []
        self.ts2 = []
        self.input_files = input_files
        self.target_files = target_files
        self.vocab_files = vocab_files
        self.batch_size = batch_size

    def readjson(self, filename):
        f = open(filename, 'r')
        ts = json.loads(f.read())
        for key in ts:
            ts_context = key['paragraphs']
            if len(ts_context) % 2 != 0:
                print('error')
                print(ts_context)
                continue
            for i in range(0, len(ts_context)):
                if (i%2==0):
                    self.ts1.append(ts_context[i])
                else:
                    self.ts2.append(ts_context[i])

    def writeinfo(self):
        f1 = open('train.1', 'w')
        f2 = open('train.2', 'w')
        for i in range(0, len(self.ts1)):
            f1.write(self.ts1[i]+'\n')
            f2.write(self.ts2[i]+'\n')


    def _init_iterator(self):
        input_dataset = tf.data.TextLineDataset(self.input_files)
        target_dataset = tf.data.TextLineDataset(self.target_files)
        input_target_dataset = tf.data.Dataset.zip((input_dataset, target_dataset))

        vocab_table = lookup_ops.index_table_from_file(self.vocab_file, default_value=UNK_ID)
        eos_id = tf.cast(src_vocab_table.lookup(tf.constant(EOS)), tf.int32)

        batched_dataset = input_target_dataset.padded_batch(
                self.batch_size,
                padded_shapes=((tf.TensorShape([None]),  # source vectors of unknown size
                    tf.TensorShape([])),     # size(source)
                    (tf.TensorShape([None]),  # target vectors of unknown size
                    tf.TensorShape([]))),    # size(target)
                padding_values=((eos_id,  # source vectors padded on the right with src_eos_id
                    0),          # size(source) -- unused
                    (eos_id,  # target vectors padded on the right with tgt_eos_id
                    0)))         # size(target) -- unused
        batched_iterator = batched_dataset.make_initializable_iterator()
        return batched_iterator







if __name__ == "__main__":
    d = data()
    d.readjson('/Users/sunguozheng3/Github/chinese-poetry/json/poet.tang.42000.json')
    d.writeinfo()



