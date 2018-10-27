import numpy as np
import random
import h5py
import os
import pickle
import hickle
import math

class TrainingData():
    def __init__(self, config, batch_size):
        
        self.batch_size = batch_size
        self.train_feats_path = config.train_feats_path
        self.densecap_train_path = config.densecap_train_path
        self.img2paragraph_path = config.img2paragraph_path
        # self.caption_labels_path = config.caption_labels_path

        self.densecap_feats, self.num_distribution, self.captions = self.load_data()

        self.size = len(self.densecap_feats)
        
        # print self.captions.shape # (14575, 6, 31)

        self.num_batch = int(self.size / self.batch_size)
        self.pointer = 0
        

    def load_data(self):

        image_paths = open(self.densecap_train_path).read().splitlines()
        imgs_names = map(lambda x: os.path.basename(x).split('.')[0], image_paths)
        
        # load feats
        # train_output_file = h5py.File(self.train_feats_path, 'r')
        # train_feats = train_output_file.get('feats') 
        # densecap_feats = np.asarray( [feats for feats in train_feats] )
        densecap_feats = hickle.load(self.train_feats_path)
        
        # load num_distribution, captions
        with open(self.img2paragraph_path, 'rb') as f:
            img2paragraph = pickle.load(f)

        num_distribution = np.array( map(lambda x: img2paragraph[x][0], imgs_names) )
        captions = np.array( map(lambda x: img2paragraph[x][1], imgs_names) )

        # caption_labels = hickle.load(self.caption_labels_path)

        # features = hickle.load(self.vgg_feats_path)

        # return densecap_feats, num_distribution, captions, caption_labels
        return densecap_feats, num_distribution, captions


    def next_batch(self):
        batch_data = {
                # "features": self.features[self.pointer: self.pointer+self.batch_size],
                "densecap_feats" : self.densecap_feats[self.pointer: self.pointer+self.batch_size],
                "num_distribution": self.num_distribution[self.pointer: self.pointer+self.batch_size],
                "captions": self.captions[self.pointer: self.pointer+self.batch_size],
                # "caption_labels": self.caption_labels[self.pointer: self.pointer+self.batch_size]
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
        # self.features = self.features[s]
        self.densecap_feats = self.densecap_feats[s]
        self.num_distribution = self.num_distribution[s]
        self.captions = self.captions[s]
        # self.caption_labels = self.caption_labels[s]

class SemiTrainingData():
    def __init__(self, train_feats_path, captions_path, batch_size):
        
        self.batch_size = batch_size
        self.train_feats_path = train_feats_path
        self.captions_path = captions_path
        # self.caption_labels_path = config.caption_labels_path

        self.densecap_feats, self.captions = self.load_data()

        self.size = len(self.densecap_feats)

        self.num_batch = int(self.size / self.batch_size)
        self.pointer = 0
        

    def load_data(self):

        # load feats
        
        densecap_feats = hickle.load(self.train_feats_path)
        captions = np.load(self.captions_path)

        # return densecap_feats, num_distribution, captions, caption_labels
        return densecap_feats, captions


    def next_batch(self):
        batch_data = {
                # "features": self.features[self.pointer: self.pointer+self.batch_size],
                "densecap_feats" : self.densecap_feats[self.pointer: self.pointer+self.batch_size],
                # "num_distribution": self.num_distribution[self.pointer: self.pointer+self.batch_size],
                "captions": self.captions[self.pointer: self.pointer+self.batch_size],
                # "caption_labels": self.caption_labels[self.pointer: self.pointer+self.batch_size]
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
        # self.features = self.features[s]
        self.densecap_feats = self.densecap_feats[s]
        # self.num_distribution = self.num_distribution[s]
        self.captions = self.captions[s]
        # self.caption_labels = self.caption_labels[s]


class ValidateData():
    def __init__(self, config, batch_size):
        
        self.batch_size = batch_size
        self.validate_feats_path = config.validate_feats_path 

        self.densecap_feats = self.load_data()
        self.size = len(self.densecap_feats)
        self.num_batch = int( math.ceil( self.size / float(self.batch_size)) )
        
        self.pointer = 0

    def load_data(self):
        
        # load feats
        densecap_feats = hickle.load(self.validate_feats_path)

        return densecap_feats


    def next_batch(self):
        batch_data = {
                # "features": self.features[self.pointer: self.pointer+self.batch_size],
                "densecap_feats" : self.densecap_feats[self.pointer: self.pointer+self.batch_size],
            }

        self.pointer = self.pointer + self.batch_size

        return batch_data

    def reset_pointer(self):
        self.pointer = 0

class TestData():
    def __init__(self, config, batch_size):
        
        self.batch_size = batch_size
        self.test_feats_path = config.test_feats_path

        self.densecap_feats = self.load_data()

        self.size = len(self.densecap_feats)
        self.num_batch = int( math.ceil( self.size / float(self.batch_size)) )
        
        self.pointer = 0

    def load_data(self):
        
        # load feats
        densecap_feats = hickle.load(self.test_feats_path)

        return densecap_feats


    def next_batch(self):
        batch_data = {
                "densecap_feats" : self.densecap_feats[self.pointer: self.pointer+self.batch_size],
            }

        self.pointer = self.pointer + self.batch_size

        return batch_data

    def reset_pointer(self):
        self.pointer = 0

        
