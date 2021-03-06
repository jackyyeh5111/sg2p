import numpy as np
import os
import random

class PathConfig():
	''' define data paths '''

	def __init__(self): 
		
		self.log_dir = self.__path('logs/')
		self.model_dir = self.__path("models/")
		self.result_dir = self.__path("results/")

		self.n_test_data_sents_path = self.__path('data/n_test_data_sents.hkl')

		self.paragraph_json_path = self.__path('data/paragraphs_v1.json')

		self.train_imgs_ids_path = self.__path('data/image_ids/train_imgs_ids.json')
		self.val_imgs_ids_path = self.__path('data/image_ids/val_imgs_ids.json')
		self.test_imgs_ids_path = self.__path('data/image_ids/test_imgs_ids.json')

		self.train_feats_path = self.__path('data/densecap/im2p_train_output.h5')
		self.val_feats_path = self.__path('data/densecap/im2p_val_output.h5')
		self.test_feats_path = self.__path('data/densecap/im2p_test_output.h5')


		self.word2idx_path = self.__path('data/word2idx.json')
		self.idx2word_path = self.__path('data/idx2word.json')
		self.img2paragraph_path = self.__path('data/img2paragraph_modify_batch')
		self.embed_matrix_path = self.__path("data/pretrained_embed_matrix.npy")

		self.reference_path = self.__path("data/reference.txt")
		self.val_reference_path = self.__path("data/val_reference.txt")

	def __path(self, target_path, base_path='..'):

		return os.path.join(base_path, target_path)
		

