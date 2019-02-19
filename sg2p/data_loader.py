import numpy as np
import random
import os
import pickle
import json
import hickle
import h5py
import math


class DataLoader(object):
    def __init__(self, mode,
                       batch_size,
                       sg_path,
                       box_feats_path,
                       max_n_objs,
                       max_n_rels,
                       max_n_attrs,
                       img_ids_path,
                       img2paragraph_path,
                       classes_1600to282,
                       use_box_feats):

        assert mode in ['train', 'val', 'test', 'sample']

        self.mode = mode
        self.batch_size = batch_size
        self.sg_path = sg_path
        self.max_n_objs = max_n_objs
        self.max_n_rels = max_n_rels
        self.max_n_attrs = max_n_attrs
        self.img_ids_path = img_ids_path
        self.img2paragraph_path = img2paragraph_path
        self.classes_1600to282 = classes_1600to282
        

        self.objs, self.triples, self.n_objs, self.attrs = self.load_graphs()

        if use_box_feats:
            self.box_feats = hickle.load(box_feats_path)

        # data to be infered must have objects
        # type of n_objs: list
        if mode in ['test']: assert 0 not in self.n_objs 

        if mode in ['train', 'sample']:
            
            # only training data requires ground truth
            self.num_distribution, self.captions = self.load_data()

            # filter entry
            keep_entry = [i for i, n_obj in enumerate(self.n_objs) if n_obj>=10 and n_obj<=self.max_n_objs]
            self.num_distribution = self.num_distribution[keep_entry]
            self.captions = self.captions[keep_entry]
            self.objs = self.objs[keep_entry]
            self.triples = self.triples[keep_entry]

            if use_box_feats:
                self.box_feats = self.box_feats[keep_entry]


        # formulate data to feed
        self.size = len(self.objs)
        self.pointer = 0
        if mode in ['train', 'sample']:
            self.data = {
                'objs': self.objs,
                'triples': self.triples,
                'attrs': self.attrs,
                'num_distribution': self.num_distribution,
                'captions': self.captions,
            }
            self.num_batch = int(self.size / self.batch_size)
            
        elif mode in ['val', 'test']:
            self.data = {
                'objs': self.objs,
                'triples': self.triples,
                'attrs': self.attrs,
            }
            self.num_batch = int( math.ceil( self.size / float(self.batch_size)) )


        if use_box_feats:
            self.data['box_feats'] = self.box_feats


    def load_data(self):

        with open(self.img_ids_path, 'r') as f:
            img_ids = json.load(f)

        # load feats
        # h5_data = h5py.File(self.train_feats_path, 'r')
        # feats = h5_data.get('feats')
        # densecap_feats = np.asarray( [feat for feat in feats] )

        # load num_distribution, captions
        with open(self.img2paragraph_path, 'rb') as f:
            img2paragraph = pickle.load(f)

        num_distribution = np.array( ([img2paragraph[str(_id)][0] for _id in img_ids]) )
        captions = np.array( ([img2paragraph[str(_id)][1] for _id in img_ids]) )

        self.img_ids = img_ids

        return num_distribution, captions

    def load_graphs(self, attr_thres=0.08):
        '''
        entries: list of dictionary
        entries[i]['labels']: shape of (O, )
        entries[i]['rels']: shape of (R, 3)
        entries[i]['boxes']: shape of (O, 4)
        entries[i]['attrs']: shape of (O, 3)
        entries[i]['attrs_conf']: shape of (O, 3)

        given an image with 4 objs detected
        outputs:
            objs: [5, 5, 4, ..., 0, ..., 0] shape of (31,) *0 is for padding
            triples: [[0, 1, 7], [0, 2, 6], ..., [31, 31, 0]] shape of (100, 3) *[31, 31, 0] s for padding
        '''

        with open(self.sg_path, 'r') as f:
            entries = pickle.load(f)

        objs = []
        attrs = []
        n_objs = []
        triples = []
        obj_count = {}
        padding = 0
        padding_attr = 400
        for i, entry in enumerate(entries):
            i_objs = [self.classes_1600to282[obj] for obj in entry['labels']]
            i_triples = entry['rels']
            i_attr = entry['attrs']
            i_attrs_conf = entry['attrs_conf']

            obj_count[len(i_objs)] = obj_count.get(len(i_objs), 0) + 1
            
            # objects
            if len(i_objs) < self.max_n_objs:
                i_objs += [padding] * (self.max_n_objs + 1 - len(i_objs)) # padding, 282 is padding idx for objs
            else:
                i_objs = i_objs[:self.max_n_objs] 
                i_objs = i_objs + [padding] # 31st idx is padding

            # relations
            if len(i_triples) < self.max_n_rels:
                pad_triples = np.zeros((self.max_n_rels-len(i_triples), 3))
                pad_triples[:, :2] = self.max_n_objs + 1
                i_triples = np.concatenate((i_triples, pad_triples), axis=0)
            else:
                i_triples = i_triples[:self.max_n_rels] 

            # attrs
            attr_pad_idx = np.where(i_attrs_conf > attr_thres)
            i_attr[attr_pad_idx] = padding_attr

            # shape of attr have to align with shape of objs 
            if len(i_attr) < self.max_n_objs:
                np_pad_attr = np.full((self.max_n_objs + 1 - len(i_attr), self.max_n_attrs), padding_attr) 
                i_attr = np.concatenate((i_attr, np_pad_attr), axis=0)
            else:
                i_attr = i_attr[:self.max_n_objs] 
                np_pad_attr = np.full((1, self.max_n_attrs), padding_attr) 
                i_attr = np.concatenate((i_attr, np_pad_attr), axis=0)

            objs.append(i_objs)
            triples.append(i_triples)
            n_objs.append(len(entry['labels']))
            attrs.append(i_attr)

        return np.array(objs), np.array(triples), n_objs, np.array(attrs)

    def next_batch(self):

        batch_data = {}
        for key, value in self.data.iteritems():
            batch_data[key] = value[self.pointer:self.pointer+self.batch_size]

        self.pointer = self.pointer + self.batch_size

        return batch_data

    def reset_pointer(self):
        self.pointer = 0
        if self.mode == 'train':
            self.shuffle()

    def shuffle(self):
        # shuffle data
        s = np.arange(self.size)
        np.random.shuffle(s)
        
        # self.densecap_feats = self.densecap_feats[s]
        for key, value in self.data.iteritems():
            self.data[key] = value[s]

