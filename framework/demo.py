# -*- coding:UTF-8 -*-
from __future__ import division
import os
import config
from model.cmpd import DetModel


C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
C.random_crop = (512, 640)
C.network = 'resnet50'


weight_path = '../output/resnet_e7_l280.hdf5'
data_path = '../data/kaist_test/kaist_test_visible'
val_data = os.listdir(data_path)

detmodel = DetModel()
detmodel.creat_model(C, val_data)
detmodel.run(C,data_path, val_data, weight_path)
