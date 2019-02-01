import numpy as np
import random
import os
import pickle
import json
# import hickle
import h5py
import math
# from util import load_paragraph

class TrainingData():
    def __init__(self, args):
        
        self.batch_size = args.batch_size
        self.train_feats_path = args.path.train_feats_path
        self.img_ids_path = args.path.train_imgs_ids_path
        self.img2paragraph_path = args.path.img2paragraph_path

        self.densecap_feats, self.num_distribution, self.captions = self.load_data()

        self.size = len(self.densecap_feats)

        self.num_batch = int(self.size / self.batch_size)
        self.pointer = 0
        

    def load_data(self):

        with open(self.img_ids_path, 'r') as f:
            img_ids = json.load(f)

        # load feats
        h5_data = h5py.File(self.train_feats_path, 'r')
        feats = h5_data.get('feats')
        densecap_feats = np.asarray( [feat for feat in feats] )

        # load num_distribution, captions
        with open(self.img2paragraph_path, 'rb') as f:
            img2paragraph = pickle.load(f)

        num_distribution = np.array( ([img2paragraph[str(_id)][0] for _id in img_ids]) )
        captions = np.array( ([img2paragraph[str(_id)][1] for _id in img_ids]) )

        return densecap_feats, num_distribution, captions


    def next_batch(self):
        batch_data = {
                "densecap_feats" : self.densecap_feats[self.pointer: self.pointer+self.batch_size],
                "num_distribution": self.num_distribution[self.pointer: self.pointer+self.batch_size],
                "captions": self.captions[self.pointer: self.pointer+self.batch_size],
            }

        self.pointer = self.pointer + self.batch_size

        return batch_data

    def reset_pointer(self):
        self.pointer = 0
        self.shuffle()

    def shuffle(self):
        # shuffle data
        s = np.arange(self.size)
        np.random.shuffle(s)
        
        self.densecap_feats = self.densecap_feats[s]
        self.num_distribution = self.num_distribution[s]
        self.captions = self.captions[s]
        


class ValidateData():
    def __init__(self, args):
        
        self.batch_size = args.test_batch_size
        self.val_feats_path = args.path.val_feats_path 

        self.densecap_feats = self.load_data()
        self.size = len(self.densecap_feats)
        self.num_batch = int( math.ceil( self.size / float(self.batch_size)) )
        
        self.pointer = 0

    def load_data(self):
        
        h5_data = h5py.File(self.val_feats_path, 'r')
        feats = h5_data.get('feats')
        densecap_feats = np.asarray( [feat for feat in feats] )
       
        return densecap_feats

    def next_batch(self):
        batch_data = {
                "densecap_feats" : self.densecap_feats[self.pointer: self.pointer+self.batch_size],
            }

        self.pointer = self.pointer + self.batch_size

        return batch_data

    def reset_pointer(self):
        self.pointer = 0

class TestData():
    def __init__(self, args):
        
        self.batch_size = args.test_batch_size
        self.test_feats_path = args.path.test_feats_path

        self.densecap_feats = self.load_data()

        self.size = len(self.densecap_feats)
        self.num_batch = int( math.ceil( self.size / float(self.batch_size)) )
        
        self.pointer = 0

    def load_data(self):

        h5_data = h5py.File(self.test_feats_path, 'r')
        feats = h5_data.get('feats')
        densecap_feats = np.asarray( [feat for feat in feats] )

        return densecap_feats


    def next_batch(self):
        batch_data = {
                "densecap_feats" : self.densecap_feats[self.pointer: self.pointer+self.batch_size],
            }

        self.pointer = self.pointer + self.batch_size

        return batch_data

    def reset_pointer(self):
        self.pointer = 0

        
