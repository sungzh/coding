

from model import Model


m = Model(
         '../../nmt/nmt_data/train.en',
         '../../nmt/nmt_data/train.vi',
         '../../nmt/nmt_data/tst2013.en',
         '../../nmt/nmt_data/tst2013.vi',
         '../../nmt/nmt_data/vocab.en.bk',
         num_units=512, layers=2, dropout=0.2,
         batch_size=128, learning_rate=0.001, output_dir='./output')

m.train(5000000)
