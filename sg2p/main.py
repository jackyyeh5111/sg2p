import argparse
import tensorflow as tf
from model import Regions_Hierarchical
import json
from data_loader import DataLoader
from solver import ParagraphSolver
import numpy as np
from path_config import PathConfig
from util import *
import os

# ex: python main.py -m train -p test -gcv_feats_dim 512
# ex: python main.py -m train -p test -gcv_feats_dim 512 -gpu 1

def load_args():
    parser = argparse.ArgumentParser()

    # system config
    parser.add_argument('-no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("-gpu", dest='gpu_id', type=str, default='0')
    parser.add_argument("-use_box_feats", action='store_true', default=False)
    parser.add_argument("-use_attrs", action='store_true', default=False)
    parser.add_argument("-use_gcv_mlayer", action='store_true', default=False, help="use gcv middle layer")
    parser.add_argument("-m", dest='mode', type=str, help="have three mode: 'train', 'infer', 'interact'.")
    parser.add_argument("-p", dest="process_name", type=str, help="process name")
    parser.add_argument("-model_name", type=str, default=None, 
                        help="The directory of save model.")
    parser.add_argument("-fixed_n_sent", action="store_true", default=True,  
                        help="generatel fixed number(S_max) of sent of a paragraph while inferencing")
    parser.add_argument("-S_max", type=int, default=6, 
                        help="max sentence number per paragraph")
    parser.add_argument("-N_max", type=int, default=30, 
                        help="max words number per sentence")
    parser.add_argument("-is_early_stop", action="store_true", default=True,  
                        help="early stopping strategy will be used in training")
    parser.add_argument("-ref_test_sents", action="store_true", default=False)
    parser.add_argument("-save_every", type=int, default=50)
    parser.add_argument("-log_every", type=int, default=50)
    parser.add_argument("-patience", type=int, default=6)
    parser.add_argument("-checkpoint", type=str, default=None, help="The directory of save model.")


    # model parameters
    parser.add_argument("-update_rule", type=str, default="adam")
    parser.add_argument("-learning_rate", type=float, default=1e-4)
    parser.add_argument("-n_epochs", type=int, default=2000)
    parser.add_argument("-sentRNN_lstm_dim", type=int, default=512)
    parser.add_argument("-wordRNN_lstm_dim", type=int, default=512)
    parser.add_argument("-num_boxes", type=int, default=50)
    parser.add_argument("-gcv_feats_dim", type=int, default=512)
    parser.add_argument("-attention_dim", type=int, default=4096)
    parser.add_argument("-project_dim", type=int, default=1024)
    parser.add_argument('-word_lstm_layer', type=int, default=1,
                        help='the num layer for word rnn')
    parser.add_argument('-sent_lstm_layer', type=int, default=1,
                        help='the num layer for sentence rnn')
    parser.add_argument('-topic_dim', type=int, default=1024,
                        help='the size for topic vector')
    parser.add_argument('-embedding_dim', type=int, default=100)
    parser.add_argument('-box_feats_dim', type=int, default=2048)
    
    parser.add_argument('-pooling_dim', type=int, default=1024,
                        help='the size for pooling vector')
    parser.add_argument('-lambda_sentence', type=int, default=5,
                        help='the cost lambda for sentence loss function')
    parser.add_argument('-lambda_word', type=int, default=1,
                        help='the cost lambda for word loss function')

    parser.add_argument('-max_n_objs', type=int, default=50)
    parser.add_argument('-max_n_rels', type=int, default=300)
    parser.add_argument('-max_n_attrs', type=int, default=3)
    parser.add_argument("-n_obj", dest='n_obj', type=int)

    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-test_batch_size", type=int, default=256)
      

    args = parser.parse_args()

    if not args.mode:  
        parser.error('mode is not given')
    
    if not args.process_name:  
        parser.error('process_name is not given')
    
    if args.mode == "infer" and not args.model_name:
        parser.error('model is not given')

    if args.n_obj == None:
        parser.error('n_obj is not given')

    return args


class DataContainer():
    def __init__(self, args):     

        self.args = args
        
        with open(args.path.word2idx_path, 'r') as f:
            self.word2idx = json.load(f)

        with open(args.path.idx2word_path, 'r') as f:
            self.idx2word = json.load(f)

        self.embed_matrix = np.load( args.path.embed_matrix_path )

        self.idx2pred, self.classes_base = loadMapDict(self.args.path.VG_SGG_dict)

        self.v2k_classes_base = {} # value to key (start from '1')
        for key, value in self.classes_base.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
            self.v2k_classes_base[value] = int(key)

        # Load classes_1600 (from bottom-up objects list)
        self.classes_1600 = []
        with open(os.path.join(self.args.path.classes_1600_path, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                self.classes_1600.append(object.split(',')[0].lower().strip())

        # map classes_1600 to classes_282
        self.classes_1600_mapping = {}
        for i, c in enumerate(self.classes_1600):
            if c in self.v2k_classes_base.keys():
                self.classes_1600_mapping[i] = self.v2k_classes_base[c]

        # Load attributes
        self.attrs = []
        with open(os.path.join(self.args.path.classes_1600_path, 'attributes_vocab.txt')) as f:
            for att in f.readlines():
                self.attrs.append(att.split(',')[0].lower().strip())

        if args.mode == "train":
            # pass
          self.train_data = self.__init_data_loader('train')
          # self.train_data = self.__init_data_loader('val')
          # self.train_data = self.__init_data_loader('sample')
          self.val_data = self.__init_data_loader('test')

        elif args.mode == "infer":
          self.test_data = TestData(batch_size=args.test_batch_size)
        
    def __init_data_loader(self, mode):
        if mode == 'train':
            batch_size = self.args.batch_size
            sg_path = self.args.path.train_sg_path
            box_feats_path = self.args.path.train_box_feats_path
            img_ids_path = self.args.path.train_imgs_ids_path
            img2paragraph_path = self.args.path.img2paragraph_path
        
        elif mode == 'val':
            batch_size = self.args.test_batch_size
            sg_path = self.args.path.val_sg_path
            box_feats_path = self.args.path.val_box_feats_path
            img_ids_path = self.args.path.val_imgs_ids_path
            img2paragraph_path = None
        
        elif mode == 'test':
            batch_size = self.args.test_batch_size
            sg_path = self.args.path.test_sg_path
            box_feats_path = self.args.path.test_box_feats_path
            img_ids_path = self.args.path.test_imgs_ids_path
            img2paragraph_path = None
        
        elif mode == 'sample':
            batch_size = self.args.test_batch_size
            sg_path = self.args.path.sample_sg_path
            box_feats_path = self.args.path.sample_box_feats_path
            img_ids_path = self.args.path.sample_imgs_ids_path
            img2paragraph_path = self.args.path.img2paragraph_path
        

        data_loader = DataLoader(  mode,
                                   batch_size,
                                   sg_path,
                                   box_feats_path,
                                   self.args.max_n_objs,
                                   self.args.max_n_rels,
                                   self.args.max_n_attrs,
                                   img_ids_path,
                                   img2paragraph_path,
                                   self.classes_1600_mapping,
                                   self.args.use_box_feats,
                                   self.args.use_attrs)

        return data_loader

def main():
    
    args = load_args()
    args.path = PathConfig(args.n_obj)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    dc = DataContainer(args)

    model = Regions_Hierarchical( word2idx = dc.word2idx, 
                                  batch_size = args.batch_size, 
                                  pretrained_embed_matrix = dc.embed_matrix, 
                                  sentRNN_lstm_dim=args.sentRNN_lstm_dim,
                                  wordRNN_lstm_dim=args.wordRNN_lstm_dim,
                                  max_n_objs=args.max_n_objs,
                                  max_n_rels=args.max_n_rels,
                                  embedding_dim=args.embedding_dim,
                                  gcv_feats_dim=args.gcv_feats_dim,
                                  use_box_feats=args.use_box_feats,
                                  use_attrs=args.use_attrs,
                                  use_gcv_mlayer=args.use_gcv_mlayer,
                                  box_feats_dim=args.box_feats_dim if args.use_box_feats else 0,
                                  n_objs=args.n_obj)


    solver = ParagraphSolver(model, dc, args)

    if args.mode == "train":
        solver.train()
    elif args.mode == "infer":
        solver.inference()
    

if __name__ == '__main__':
    main()
