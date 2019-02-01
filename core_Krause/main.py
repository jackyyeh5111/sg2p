import argparse
import tensorflow as tf
from model import Regions_Hierarchical
import json
from data_loader import TrainingData, ValidateData, TestData
from solver import ParagraphSolver
import numpy as np
from path_config import PathConfig

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_args():
    parser = argparse.ArgumentParser()

    # system config
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("-m", dest='mode', type=str, help="have three mode: 'train', 'infer', 'interact'.")
    parser.add_argument("-p", dest="process_name", type=str, help="process name")
    parser.add_argument("--model_name", type=str, default=None, 
                        help="The directory of save model.")
    parser.add_argument("--fixed_n_sent", action="store_true", default=False,  
                        help="generatel fixed number(S_max) of sent of a paragraph while inferencing")
    parser.add_argument("--S_max", type=int, default=6, 
                        help="max sentence number per paragraph")
    parser.add_argument("--N_max", type=int, default=30, 
                        help="max words number per sentence")
    parser.add_argument("--early_stop", action="store_true", default=False,  
                        help="early stopping strategy will be used in training")
    parser.add_argument("--ref_test_sents", action="store_true", default=False)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None, help="The directory of save model.")

    # model parameters
    parser.add_argument("--update_rule", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=600)
    parser.add_argument("--sentRNN_lstm_dim", type=int, default=512)
    parser.add_argument("--wordRNN_lstm_dim", type=int, default=512)
    parser.add_argument("--num_boxes", type=int, default=50)
    parser.add_argument("--feats_dim", type=int, default=4096)
    parser.add_argument("--attention_dim", type=int, default=4096)
    parser.add_argument("--project_dim", type=int, default=1024)
    parser.add_argument('--word_lstm_layer', type=int, default=1,
                        help='the num layer for word rnn')
    parser.add_argument('--sent_lstm_layer', type=int, default=1,
                        help='the num layer for sentence rnn')
    parser.add_argument('--topic_dim', type=int, default=1024,
                        help='the size for topic vector')
    parser.add_argument('--pooling_dim', type=int, default=1024,
                        help='the size for pooling vector')
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--lambda_sentence', type=int, default=5,
                        help='the cost lambda for sentence loss function')
    parser.add_argument('--lambda_word', type=int, default=1,
                        help='the cost lambda for word loss function')

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
      

    args = parser.parse_args()

    if not args.mode:  
        parser.error('mode is not given')
    
    if not args.process_name:  
        parser.error('process_name is not given')
    
    if args.mode == "infer" and not args.model_name:
        parser.error('model is not given')

    return args


class Data():
    def __init__(self, args):     

        self.args = args
        
        with open(args.path.word2idx_path, 'r') as f:
            self.word2idx = json.load(f)

        with open(args.path.idx2word_path, 'r') as f:
            self.idx2word = json.load(f)

        self.embed_matrix = np.load( args.path.embed_matrix_path )

        if args.mode == "train":
            # pass
          self.train_data = TrainingData(args)
          self.val_data = TestData(args)

        # else args.mode == "infer":
        #   self.test_data = TestData(batch_size=args.test_batch_size, self.id2paragraph)
        

def main():
    
    args = load_args()
    args.path = PathConfig()

    data = Data(args)

    model = Regions_Hierarchical( word2idx = data.word2idx, 
                                  batch_size = args.batch_size, 
                                  pretrained_embed_matrix = data.embed_matrix, 
                                  sentRNN_lstm_dim=args.sentRNN_lstm_dim,
                                  wordRNN_lstm_dim=args.wordRNN_lstm_dim)


    solver = ParagraphSolver(model, data, args)

    if args.mode == "train":
        solver.train()
    elif args.mode == "infer":
        solver.inference()
    

if __name__ == '__main__':
    main()
