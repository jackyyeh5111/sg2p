import numpy as np
import os
import random

class PathConfig():
	''' define data paths '''

	def __init__(self): 
		
		self.log_dir = self.__path('logs/')
		self.model_dir = self.__path("models/")
		self.result_dir = self.__path("results/")

		self.image_dir = '../../dataset/VG'
		
		self.n_test_data_sents_path = self.__path('data/n_test_data_sents.hkl')

		self.paragraph_json_path = self.__path('data/paragraphs_v1.json')

		self.train_imgs_ids_path = self.__path('data/image_ids/train_imgs_ids.json')
		self.val_imgs_ids_path = self.__path('data/image_ids/val_imgs_ids.json')
		self.test_imgs_ids_path = self.__path('data/image_ids/test_imgs_ids.json')
		self.sample_imgs_ids_path = self.__path('data/image_ids/sample_imgs_ids.json')

		self.train_feats_path = self.__path('data/densecap/im2p_train_output.h5')
		self.val_feats_path = self.__path('data/densecap/im2p_val_output.h5')
		self.test_feats_path = self.__path('data/densecap/im2p_test_output.h5')

		self.train_box_feats_path = self.__path('data/sg/box_features_train.hkl')
		self.val_box_feats_path = self.__path('data/sg/box_features_val.hkl')
		self.test_box_feats_path = self.__path('data/sg/box_features_test.hkl')
		self.sample_box_feats_path = self.__path('data/sg/box_features_sample.hkl')

		self.train_sg_path = self.__path('data/sg/sg2p_train.pkl')
		self.val_sg_path = self.__path('data/sg/sg2p_val.pkl')
		self.test_sg_path = self.__path('data/sg/sg2p_test.pkl')
		self.sample_sg_path = self.__path('data/sg/sg2p_sample.pkl')

		self.VG_SGG_dict = self.__path('data/sg/VG-SGG-dicts_jacky.json')
		# self.classes_path = '/2t_1/jackyyeh/scene-graph-TF-release/data_tools/VG/jacky_object_list.txt'
		self.classes_1600_path = '../../bottom-up-attention/data/genome/1600-400-20' # 1600 objects, 400 attributes, 20 relations


		self.word2idx_path = self.__path('data/word2idx.json')
		self.idx2word_path = self.__path('data/idx2word.json')
		self.img2paragraph_path = self.__path('data/img2paragraph_modify_batch')
		self.embed_matrix_path = self.__path("data/pretrained_embed_matrix.npy")

		self.reference_path = self.__path("data/reference.txt")
		self.val_reference_path = self.__path("data/val_reference.txt")

	def __path(self, target_path, base_path='..'):

		return os.path.join(base_path, target_path)
		

