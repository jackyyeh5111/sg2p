import numpy as np
import random
import os
import pickle
import json
import hickle
import h5py
import math
# from util import load_paragraph

def combineData(train_data, val_data, args):

    use_box_feats = args.use_box_feats
    use_attrs = args.use_attrs
    spt_feats = args.spt_feats
    
    train_data.objs = np.concatenate((train_data.objs, val_data.objs), axis=0)
    train_data.triples = np.concatenate((train_data.triples, val_data.triples), axis=0)
    train_data.num_distribution = np.concatenate((train_data.num_distribution, val_data.num_distribution), axis=0)
    train_data.captions = np.concatenate((train_data.captions, val_data.captions), axis=0)
    
    if use_box_feats:
        train_data.box_feats = np.concatenate((train_data.box_feats, val_data.box_feats), axis=0)
        if spt_feats:
            train_data.rel_feats = np.concatenate((train_data.rel_feats, val_data.rel_feats), axis=0)

    # if use_box_feats:
    #     train_data.box_feats = np.concatenate((train_data.box_feats, val_data.box_feats), axis=0)

    if use_attrs:
        train_data.attrs = np.concatenate((train_data.attrs, val_data.attrs), axis=0)

    train_data.size = len(train_data.captions)

    train_data.num_batch = int(train_data.size / train_data.batch_size)    
 
class TrainingData():
    def __init__(self, args, classes_1600to282, mode):
        
        assert mode in ['train', 'val', 'sample']

        self.batch_size = args.batch_size
        self.max_n_objs = args.max_n_objs
        self.max_n_rels = args.max_n_rels
        self.use_box_feats = args.use_box_feats
        self.use_attrs = args.use_attrs
        self.spt_feats = args.spt_feats
        self.max_n_attrs = args.max_n_attrs
        
        if mode == 'train':
            self.sg_path = args.path.train_sg_path
            self.box_feats_path = args.path.train_box_feats_path
            self.rel_feats_path = args.path.train_rel_feats_path
            self.img_ids_path = args.path.train_imgs_ids_path
        if mode == 'val':
            self.sg_path = args.path.val_sg_path
            self.box_feats_path = args.path.val_box_feats_path
            self.rel_feats_path = args.path.train_rel_feats_path
            self.img_ids_path = args.path.val_imgs_ids_path
        if mode == 'sample':
            self.sg_path = args.path.sample_sg_path
            self.box_feats_path = args.path.sample_box_feats_path
            self.rel_feats_path = args.path.train_rel_feats_path
            self.img_ids_path = args.path.sample_imgs_ids_path

        
        self.img2paragraph_path = args.path.img2paragraph_path

        self.num_distribution, self.captions = self.load_data()


        self.objs, self.triples, self.n_objs, self.attrs = self.load_graphs(self.sg_path, 
                                                                classes_1600to282, 
                                                                self.max_n_objs, 
                                                                self.max_n_rels,
                                                                self.max_n_attrs)
        
        if self.use_box_feats:
            self.box_feats = hickle.load(self.box_feats_path)
            if self.spt_feats:
                self.rel_feats = hickle.load(self.rel_feats_path)


        keep_entry = [i for i, n_obj in enumerate(self.n_objs) if n_obj>=10 and n_obj<=self.max_n_objs]

        # filter entry
        self.num_distribution = self.num_distribution[keep_entry]
        self.captions = self.captions[keep_entry]
        self.objs = self.objs[keep_entry]
        self.triples = self.triples[keep_entry]

        if self.use_box_feats:
            self.box_feats = self.box_feats[keep_entry]
            if self.spt_feats:
                self.rel_feats = self.rel_feats[keep_entry]
        
        print 'rel_feats ok'


        if self.use_attrs:
            self.attrs = self.attrs[keep_entry]

        self.size = len(self.captions)

        self.num_batch = int(self.size / self.batch_size)
        self.pointer = 0
        
    def load_graphs(self, sg_path, classes_1600to282, max_n_objs, max_n_rels, max_n_attrs, attr_thres=0.08):
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

        with open(sg_path, 'r') as f:
            entries = pickle.load(f)

        assert len(entries) == len(self.captions)

        objs = []
        attrs = []
        n_objs = []
        triples = []
        obj_count = {}

        n = []
        padding = 0
        padding_attr = 400
        for i, entry in enumerate(entries):
            # i_objs_idx = list(entry['labels'])
            i_objs = [classes_1600to282[obj] for obj in entry['labels']]
            i_triples = entry['rels']
            i_triples = entry['rels']
            i_attr = entry['attrs']
            i_attrs_conf = entry['attrs_conf']


            obj_count[len(i_objs)] = obj_count.get(len(i_objs), 0) + 1
            
            if len(i_objs) < max_n_objs:
                # i_objs_idx += [-1] * (max_n_objs-len(i_objs)) # included in triples, no need to feed into model
                i_objs += [padding] * (max_n_objs + 1 - len(i_objs)) # padding, 282 is padding idx for objs
            else:
                i_objs = i_objs[:max_n_objs] + [padding] # 31st idx is padding


            if len(i_triples) < max_n_rels:
                pad_triples = np.zeros((max_n_rels-len(i_triples), 3))
                pad_triples[:, :2] = max_n_objs + 1
                i_triples = np.concatenate((i_triples, pad_triples), axis=0)
            else:
                i_triples = i_triples[:max_n_rels] 

            # attrs
            attr_pad_idx = np.where(i_attrs_conf < attr_thres)
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


    def next_batch(self):
        batch_data = {
                "objs" : self.objs[self.pointer: self.pointer+self.batch_size],
                "triples" : self.triples[self.pointer: self.pointer+self.batch_size],
                "num_distribution": self.num_distribution[self.pointer: self.pointer+self.batch_size],
                "captions": self.captions[self.pointer: self.pointer+self.batch_size],
            }

        if self.use_box_feats:
            batch_data['box_feats'] = self.box_feats[self.pointer: self.pointer+self.batch_size]
            if self.spt_feats:
                batch_data['rel_feats'] = self.rel_feats[self.pointer: self.pointer+self.batch_size]

        if self.use_attrs:
            batch_data['attrs'] = self.attrs[self.pointer: self.pointer+self.batch_size]

        self.pointer = self.pointer + self.batch_size

        return batch_data

    def reset_pointer(self):
        self.pointer = 0
        self.shuffle()

    def shuffle(self):
        # shuffle data
        s = np.arange(self.size)
        np.random.shuffle(s)
        
        # self.densecap_feats = self.densecap_feats[s]
        self.objs = self.objs[s]
        self.triples = self.triples[s]
        self.num_distribution = self.num_distribution[s]
        self.captions = self.captions[s]

        if self.use_box_feats:
            self.box_feats = self.box_feats[s]
            if self.spt_feats:
                self.rel_feats = self.rel_feats[s]

        if self.use_attrs:
            self.attrs = self.attrs[s]


class TestData():
    def __init__(self, args, classes_1600to282):
        
        self.batch_size = args.test_batch_size
        self.test_feats_path = args.path.test_feats_path
        self.test_sg_path = args.path.test_sg_path
        self.max_n_objs = args.max_n_objs
        self.max_n_rels = args.max_n_rels
        self.box_feats_path = args.path.test_box_feats_path
        self.max_n_attrs = args.max_n_attrs
        self.use_box_feats = args.use_box_feats
        self.use_attrs = args.use_attrs
        self.spt_feats = args.spt_feats
        self.rel_feats_path = args.path.test_rel_feats_path

        # self.densecap_feats = self.load_data()
        self.objs, self.triples, self.n_objs, self.attrs = self.load_graphs(self.test_sg_path, 
                                                                classes_1600to282, 
                                                                self.max_n_objs, 
                                                                self.max_n_rels,
                                                                self.max_n_attrs)

        if self.use_box_feats:
            self.box_feats = hickle.load(self.box_feats_path)
            if self.spt_feats:
                self.rel_feats = hickle.load(self.rel_feats_path)


        self.size = len(self.triples)
        self.num_batch = int( math.ceil( self.size / float(self.batch_size)) )
        
        self.pointer = 0

    def load_graphs(self, sg_path, classes_1600to282, max_n_objs, max_n_rels, max_n_attrs, attr_thres=0.08):
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

        with open(sg_path, 'r') as f:
            entries = pickle.load(f)

        objs = []
        attrs = []
        n_objs = []
        triples = []
        obj_count = {}

        n = []
        padding = 0
        padding_attr = 400
        for i, entry in enumerate(entries):
            # i_objs_idx = list(entry['labels'])
            i_objs = [classes_1600to282[obj] for obj in entry['labels']]
            i_triples = entry['rels']
            i_triples = entry['rels']
            i_attr = entry['attrs']
            i_attrs_conf = entry['attrs_conf']


            obj_count[len(i_objs)] = obj_count.get(len(i_objs), 0) + 1
            
            if len(i_objs) < max_n_objs:
                # i_objs_idx += [-1] * (max_n_objs-len(i_objs)) # included in triples, no need to feed into model
                i_objs += [padding] * (max_n_objs + 1 - len(i_objs)) # padding, 282 is padding idx for objs
            else:
                i_objs = i_objs[:max_n_objs] + [padding] # 31st idx is padding


            if len(i_triples) < max_n_rels:
                pad_triples = np.zeros((max_n_rels-len(i_triples), 3))
                pad_triples[:, :2] = max_n_objs + 1
                i_triples = np.concatenate((i_triples, pad_triples), axis=0)
            else:
                i_triples = i_triples[:max_n_rels] 


            # attrs
            attr_pad_idx = np.where(i_attrs_conf < attr_thres)
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


    def load_data(self):

        h5_data = h5py.File(self.test_feats_path, 'r')
        feats = h5_data.get('feats')
        densecap_feats = np.asarray( [feat for feat in feats] )

        return densecap_feats


    def next_batch(self):
        batch_data = {
                "objs" : self.objs[self.pointer: self.pointer+self.batch_size],
                "triples" : self.triples[self.pointer: self.pointer+self.batch_size],
            }

        if self.use_attrs:
            batch_data['attrs'] = self.attrs[self.pointer: self.pointer+self.batch_size]

        if self.use_box_feats:
            batch_data['box_feats'] = self.box_feats[self.pointer: self.pointer+self.batch_size]
            if self.spt_feats:
                batch_data['rel_feats'] = self.rel_feats[self.pointer: self.pointer+self.batch_size]

        self.pointer = self.pointer + self.batch_size

        return batch_data

    def reset_pointer(self):
        self.pointer = 0

        
