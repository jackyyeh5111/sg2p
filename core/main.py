import tensorflow as tf
from model import Regions_Hierarchical
from config import Config
import json
from data_loader import TrainingData, ValidateData, TestData
from solver import ParagraphSolver
from optparse import OptionParser
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_opts():
    op = OptionParser()

    op.add_option("-m", dest="mode", type=str,  
                  help="have three mode: 'train', 'infer', 'interact'.")
    op.add_option("--process_name",
                  dest="process_name", type=str,
                  help="process name")
    op.add_option("--model_name",
                  dest="model_name", type=str, default=None,
                  help="The directory of save model.")
    op.add_option("--fixed_n_sent",
                  dest="fixed_n_sent", action="store_true", default=False, 
                  help="generate fixed number(S_max) of sent of a paragraph while inferencing")
    op.add_option("--add_multi_label",
                  dest="add_multi_label", action="store_true", default=False)
    op.add_option("--S_max",
                  dest="S_max", type=int, default=6,
                  help="max sentence number in a paragraph")
    op.add_option("--early_stop",
                  dest="early_stop", action="store_true", default=False, 
                  help="early stopping strategy will be used in training")
    op.add_option("--transfered_model_name",
                  dest="transfered_model_name", type=str, default=None,
                  help="The directory of pretrained model or transfer parameters.")
    

    op.add_option("--sentRNN_lstm_dim",
                  dest="sentRNN_lstm_dim", type=int, default=512)
    op.add_option("--wordRNN_lstm_dim",
                  dest="wordRNN_lstm_dim", type=int, default=512)
    

    (opts, args) = op.parse_args()
    if not opts.mode:  op.error('mode is not given')
    if not opts.process_name:  op.error('process_name is not given')
    if opts.mode == "infer": 
        if not opts.model_name:  op.error('model is not given')

    return opts

class Data():
    def __init__(self, config, mode):     
        
        with open(config.word2index_path, 'r') as f:
            self.word2idx = json.load(f)

        with open(config.index2word_path, 'r') as f:
            self.idx2word = json.load(f)


        self.embed_matrix = np.load(config.pretrained_embed_matrix_path)

        if mode == "train":
            self.train_data = TrainingData(config, batch_size=config.batch_size)
            self.val_data = ValidateData(config, batch_size=config.test_batch_size)

        elif mode == "infer":
            self.test_data = TestData(config, batch_size=config.test_batch_size)


def main():
    
    opts = load_opts()

    config = Config()
    data = Data(config, opts.mode)

    model = Regions_Hierarchical( word2idx = data.word2idx, 
                                  batch_size = config.batch_size, 
                                  pretrained_embed_matrix = data.embed_matrix, 
                                  sentRNN_lstm_dim=opts.sentRNN_lstm_dim,
                                  wordRNN_lstm_dim=opts.wordRNN_lstm_dim,
                                  S_max = opts.S_max)

    solver = ParagraphSolver(model, data, config, opts)

    if opts.mode == "train":
        solver.train()
    elif opts.mode == "infer":
        solver.inference()
    

if __name__ == '__main__':
    main()
