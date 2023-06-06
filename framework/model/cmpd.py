# -*- coding:UTF-8 -*-
from .base_model_conf import Base_model
from keras.optimizers import Adam
from keras.models import Model
from parallel_model import ParallelModel
from keras.utils import generic_utils
import losses as losses
import bbox_process, bbox_transform
from . import model_AP_IAFA_conf
import time, os, cv2
import config
C = config.Config()
from keras.layers import *
import random
from data_augment import _saturation_kaist
from model.MCAF.MCAF import MCAF
import scipy.io as io
import numpy as np

def one_result(path):
    matr = io.loadmat(path)
    result = matr['res'][:, 0][0][3]

    thresholds = result[:, 0]
    recall = result[:, 1]
    precision = result[:, 2]

    dict = {}
    dict['prec'] = precision
    dict['rec'] = recall
    dict['thresholds'] = thresholds

    return dict


class DetModel(Base_model):
	def name(self):
		return 'Model_2step'
	def initialize(self, opt):
		Base_model.initialize(self,opt)
		# specify the training details
		self.cls_loss_r1 = []
		self.regr_loss_r1 = []
		self.cls_loss_r2 = []
		self.regr_loss_r2 = []
		self.illuminate_loss = []
		self.losses = np.zeros((self.epoch_length, 9))
		self.optimizer = Adam(lr=opt.init_lr)
		print ('Initializing the {}'.format(self.name()))

	def creat_model(self,opt,train_data, phase='inference', augment=False):
		Base_model.create_base_model(self, opt,train_data, phase=phase, augment=augment)
		alf1, alf2, conf, bf1, bf2, conf_b = model_AP_IAFA_conf.create_AP_IAFA(self.base_layers,self.num_anchors, trainable=False)
		
		if phase=='train':
			self.model_1st = Model([self.img_input_rgb,self.img_input_lwir], alf1 + alf2 + bf1 + bf2)
			self.model_2nd = Model([self.img_input_rgb,self.img_input_lwir], [conf]+conf_b)

			self.model_2nd.compile(optimizer=self.optimizer, loss=[losses.mse_loss, losses.mse_loss, losses.mse_loss],sample_weight_mode=None)
		self.model_all = Model([self.img_input_rgb,self.img_input_lwir], alf1+alf2+[conf]+conf_b+bf1+bf2)
	

	def run(self,opt, data_path,val_data, weight_path, step=0):
		self.model_all.load_weights(weight_path, by_name=True)
		print ('load weights from {}'.format(weight_path))

		all_detectors_names = list(['rgb', 'ir', 'fusion'])
		self.mcaf = MCAF(all_detectors_names)

		for f in range(len(val_data)):

			img_name = os.path.join(data_path,val_data[f])
			if not img_name.lower().endswith(('.jpg', '.png')):
				continue
			print(img_name)
			img = cv2.imread(img_name)

			img_name_lwir = os.path.join(data_path[:-7]+'lwir', val_data[f][:-11]+'lwir.png')
			img_lwir = cv2.imread(img_name_lwir)
			
			start_time = time.time()

			bounding_boxes = {}
			labels = {}
			class_scores = {}
			uncertainty = {}
		
			x_in = bbox_process.format_img(img, opt)
			x_in_lwir = bbox_process.format_img(img_lwir, opt)
			Y = self.model_all.predict([x_in,x_in_lwir])


			proposals = bbox_process.pred_pp_1st(self.anchors, Y[0], Y[1][:,:,:4], opt)
			bbx, scores, var = bbox_process.pred_det_var(proposals, Y[2], Y[3][:,:,:4], Y[4][0], opt, step=2)
			bounding_boxes['fusion'] = np.array(bbx)
			labels['fusion'] = np.array(np.ones(bbx.shape[0]))
			class_scores['fusion'] = np.array(scores[:, 0])
			uncertainty['fusion'] = np.array(var[:, 0])


			proposals = bbox_process.pred_pp_1st(self.anchors, Y[7], Y[8][:,:,:4], opt)
			bbx, scores, var = bbox_process.pred_det_var(proposals, Y[11], Y[12][:,:,:4], Y[5][0], opt, step=2)
			bounding_boxes['rgb'] = np.array(bbx)
			labels['rgb'] = np.array(np.ones(bbx.shape[0]))
			class_scores['rgb'] = np.array(scores[:, 0])
			uncertainty['rgb'] = np.array(var[:, 0])

		
			proposals = bbox_process.pred_pp_1st(self.anchors, Y[9], Y[10][:,:,:4], opt)
			bbx, scores, var = bbox_process.pred_det_var(proposals, Y[13], Y[14][:,:,:4], Y[6][0], opt, step=2)
			bounding_boxes['ir'] = np.array(bbx)
			labels['ir'] = np.array(np.ones(bbx.shape[0]))
			class_scores['ir'] = np.array(scores[:, 0])
			uncertainty['ir'] = np.array(var[:, 0])


			bounding_box, labels, class_score, uncertainty = self.mcaf.MCAF_result(bounding_boxes, class_scores, labels, uncertainty)
			print ('Test time: %.4f s' % (time.time() - start_time))

			image_name_save,png=val_data[f].split('.')
			image_name_save = image_name_save[:-8]#lwir

			result_path =  '../result/det/'
			if not os.path.exists(result_path):
				os.makedirs(result_path)

			image_set_file = os.path.join(result_path, image_name_save + '.txt')
			list_file = open(image_set_file, 'w')
			for i in range(len(bounding_box)):
				image_write_txt = 'person' + ' ' + str(np.round(bounding_box[i][0], 4)) + ' ' + str(np.round(bounding_box[i][1], 4)) + ' ' \
								  + str(np.round(bounding_box[i][2], 4)) + ' ' + str(np.round(bounding_box[i][3], 4)) + ' ' + str(round(float(class_score[i]), 8))
				list_file.write(image_write_txt)
				list_file.write('\n')
			list_file.close()


