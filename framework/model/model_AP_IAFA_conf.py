# -*- coding:UTF-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras.layers import *
from keras import backend as K
import numpy as np
import math
from .deform_layers import ConvOffset2D
import tensorflow as tf
from .FixedBatchNormalization import FixedBatchNormalization

def prior_probability(probability=0.01):
	def f(shape, dtype=K.floatx()):
		# set bias to -log((1 - p)/p) for foregound
		result = np.ones(shape, dtype=dtype) * -math.log((1 - probability) / probability)
		return result
	return f

def AP(input,num_anchors,name,filters=256,kersize=(3,3),trainable=True):
    x_class_mean = Convolution2D(num_anchors, (3, 3),activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability(),
                            name=name+'_rpn_classnew', padding='same', trainable=trainable)(input)
    x_class_mean_reshape = Reshape((-1, 1), name=name+'_class_reshapenew')(x_class_mean)

    x_regr = Convolution2D(num_anchors * 4, (3, 3), activation='linear', kernel_initializer='glorot_normal',
                           name=name+'_rpn_regressnew', padding='same', trainable=trainable)(input)
    x_regr_reshape = Reshape((-1,4), name=name+'_regress_reshapenew')(x_regr)
   
    return x_class_mean_reshape, x_regr_reshape

def IAFA(input, num_anchors,name,filters=256,kersize=(3,3),trainable=True, channel_num=96):

    x = Convolution2D(channel_num, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conv_all_new', trainable=trainable)(input)
   
    # modality alignment
    x = ConvOffset2D(channel_num, name=name + 'deformconv_offset', trainable=trainable)(x)

    x_regr = Convolution2D(num_anchors * 4, (3, 3),padding='same', activation='linear', kernel_initializer='glorot_normal',
                           name=name+'_rpn_regress_new',trainable=trainable)(x)
    x_regr_reshape = Reshape((-1,4), name=name+'_regress_reshape')(x_regr)

    x_regr_var = Convolution2D(num_anchors * 4, (3, 3),padding='same', activation='linear', kernel_initializer='glorot_normal',
                           name=name+'_rpn_regress_var_new',trainable=trainable)(x)
    x_regr_var_reshape = Reshape((-1,4), name=name+'_regress_var_reshape')(x_regr_var)

    x_regr_with_var = Concatenate()([x_regr_reshape, x_regr_var_reshape])

    x_class_mean = Convolution2D(num_anchors, (3, 3), padding='same',activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability(),
                            name=name+'_rpn_class',trainable=trainable)(x)
    x_class_mean_reshape = Reshape((-1, 1), name=name+'_class_reshape')(x_class_mean)

    x = Convolution2D(64, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conf_conv0_0', trainable=True)(x)
    x = Convolution2D(64, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conf_conv0_1', trainable=True)(x)
    x = Convolution2D(400, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conf_conv1', trainable=True)(x)
    x = Convolution2D(120, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conf_conv2_0', trainable=True)(x)
    x = Convolution2D(96, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conf_conv2_1', trainable=True)(x)
    x = Convolution2D(32, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conf_conv3_0', trainable=True)(x)
    x = Convolution2D(32, kersize, padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name=name + '_conf_conv3_1', trainable=True)(x)
    conf = Convolution2D(num_anchors * 2, kersize, padding='same', activation='sigmoid',
                      kernel_initializer='glorot_normal', name=name + '_conf_conv4', trainable=True)(x)
    conf_reshape = Reshape((-1, 2), name=name+'_conf_reshape')(conf)

    return x_class_mean_reshape, x_regr_with_var, conf_reshape

def AP_stage(AP_CONV,base_layers,num_anchors,filters=256,kersize=(3,3), trainable=True):
    # stage3 = base_layers[0]
    # stage4 = base_layers[3]
    # stage5 = base_layers[6]
    # stage6 = base_layers[9]
    stage3 = AP_CONV[0]
    stage4 = AP_CONV[1]
    stage5 = AP_CONV[2]
    stage6 = AP_CONV[3]

    P3_cls, P3_regr  = AP(stage3, num_anchors[0], name='pred0_1_base', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls, P4_regr = AP(stage4, num_anchors[1], name='pred1_1_base', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls, P5_regr  = AP(stage5, num_anchors[2], name='pred2_1_base', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls, P6_regr = AP(stage6, num_anchors[3], name='pred3_1_base', filters=filters, kersize=kersize, trainable=trainable)
    y_cls = Concatenate(axis=1, name='mbox_cls_1')([P3_cls, P4_cls, P5_cls, P6_cls])
    y_regr = Concatenate(axis=1, name='mbox_regr_1')([P3_regr, P4_regr, P5_regr, P6_regr])
    return [y_cls, y_regr]


def AB_stage(AP_CONV,base_layers,num_anchors,filters=256,kersize=(3,3), trainable=True):
    # stage3 = base_layers[0]
    # stage4 = base_layers[3]
    # stage5 = base_layers[6]
    # stage6 = base_layers[9]
    stage3_rgb = AP_CONV[0]
    stage4_rgb = AP_CONV[1]
    stage5_rgb = AP_CONV[2]
    stage6_rgb = AP_CONV[3]

    stage3_ir = AP_CONV[4]
    stage4_ir = AP_CONV[5]
    stage5_ir = AP_CONV[6]
    stage6_ir = AP_CONV[7]

    P3_cls_rgb, P3_regr_rgb  = AP(stage3_rgb, num_anchors[0], name='pred0_1_rgb', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls_rgb, P4_regr_rgb = AP(stage4_rgb, num_anchors[1], name='pred1_1_rgb', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls_rgb, P5_regr_rgb  = AP(stage5_rgb, num_anchors[2], name='pred2_1_rgb', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls_rgb, P6_regr_rgb = AP(stage6_rgb, num_anchors[3], name='pred3_1_rgb', filters=filters, kersize=kersize, trainable=trainable)

    P3_cls_ir, P3_regr_ir  = AP(stage3_ir, num_anchors[0], name='pred0_1_ir', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls_ir, P4_regr_ir = AP(stage4_ir, num_anchors[1], name='pred1_1_ir', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls_ir, P5_regr_ir  = AP(stage5_ir, num_anchors[2], name='pred2_1_ir', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls_ir, P6_regr_ir = AP(stage6_ir, num_anchors[3], name='pred3_1_ir', filters=filters, kersize=kersize, trainable=trainable)

    y_cls_rgb = Concatenate(axis=1, name='mbox_cls_rgb')([P3_cls_rgb, P4_cls_rgb, P5_cls_rgb, P6_cls_rgb])
    y_regr_rgb = Concatenate(axis=1, name='mbox_regr_rgb')([P3_regr_rgb, P4_regr_rgb, P5_regr_rgb, P6_regr_rgb])

    y_cls_ir = Concatenate(axis=1, name='mbox_cls_ir')([P3_cls_ir, P4_cls_ir, P5_cls_ir, P6_cls_ir])
    y_regr_ir = Concatenate(axis=1, name='mbox_regr_ir')([P3_regr_ir, P4_regr_ir, P5_regr_ir, P6_regr_ir])
    return [y_cls_rgb, y_regr_rgb, y_cls_ir, y_regr_ir]

def AB_stage2(AP_CONV,base_layers,num_anchors,filters=256,kersize=(3,3), trainable=True):
    stage3_rgb = AP_CONV[0]
    stage4_rgb = AP_CONV[1]
    stage5_rgb = AP_CONV[2]
    stage6_rgb = AP_CONV[3]

    stage3_ir = AP_CONV[4]
    stage4_ir = AP_CONV[5]
    stage5_ir = AP_CONV[6]
    stage6_ir = AP_CONV[7]

    P3_cls_rgb, P3_regr_rgb, P3_conf_rgb  = IAFA(stage3_rgb, num_anchors[0], name='pred0_2_rgb', filters=filters, kersize=kersize, trainable=trainable, channel_num=48)
    P4_cls_rgb, P4_regr_rgb, P4_conf_rgb = IAFA(stage4_rgb, num_anchors[1], name='pred1_2_rgb', filters=filters, kersize=kersize, trainable=trainable, channel_num=48)
    P5_cls_rgb, P5_regr_rgb, P5_conf_rgb  = IAFA(stage5_rgb, num_anchors[2], name='pred2_2_rgb', filters=filters, kersize=kersize, trainable=trainable, channel_num=48)
    P6_cls_rgb, P6_regr_rgb, P6_conf_rgb = IAFA(stage6_rgb, num_anchors[3], name='pred3_2_rgb', filters=filters, kersize=kersize, trainable=trainable, channel_num=48)

    P3_cls_ir, P3_regr_ir, P3_conf_ir  = IAFA(stage3_ir, num_anchors[0], name='pred0_2_ir', filters=filters, kersize=kersize, trainable=trainable, channel_num=48)
    P4_cls_ir, P4_regr_ir, P4_conf_ir = IAFA(stage4_ir, num_anchors[1], name='pred1_2_ir', filters=filters, kersize=kersize, trainable=trainable, channel_num=48)
    P5_cls_ir, P5_regr_ir, P5_conf_ir  = IAFA(stage5_ir, num_anchors[2], name='pred2_2_ir', filters=filters, kersize=kersize, trainable=trainable, channel_num=48)
    P6_cls_ir, P6_regr_ir, P6_conf_ir = IAFA(stage6_ir, num_anchors[3], name='pred3_2_ir', filters=filters, kersize=kersize, trainable=trainable, channel_num=48)

    y_cls_rgb = Concatenate(axis=1, name='mbox_cls_rgb2')([P3_cls_rgb, P4_cls_rgb, P5_cls_rgb, P6_cls_rgb])
    y_regr_rgb = Concatenate(axis=1, name='mbox_regr_rgb2')([P3_regr_rgb, P4_regr_rgb, P5_regr_rgb, P6_regr_rgb])
    conf_pred_rgb = Concatenate(axis=1, name='mbox_conf_rgb2')([P3_conf_rgb, P4_conf_rgb, P5_conf_rgb, P6_conf_rgb])


    y_cls_ir = Concatenate(axis=1, name='mbox_cls_ir2')([P3_cls_ir, P4_cls_ir, P5_cls_ir, P6_cls_ir])
    y_regr_ir = Concatenate(axis=1, name='mbox_regr_ir2')([P3_regr_ir, P4_regr_ir, P5_regr_ir, P6_regr_ir])
    conf_pred_ir = Concatenate(axis=1, name='mbox_conf_ir2')([P3_conf_ir, P4_conf_ir, P5_conf_ir, P6_conf_ir])
    return [y_cls_rgb, y_regr_rgb, y_cls_ir, y_regr_ir], [conf_pred_rgb, conf_pred_ir]



def IAFA_stage(AP_CONV,base_layers, num_anchors, filters=256, kersize=(3,3),trainable=True):
    # stage3 = base_layers[0]
    stage3 = AP_CONV[0]
    stage3_rgb = base_layers[1]
    stage3_lwir = base_layers[2]
    # stage4 = base_layers[3]
    stage4 = AP_CONV[1]
    stage4_rgb = base_layers[4]
    stage4_lwir = base_layers[5]
    # stage5 = base_layers[6]
    stage5 =  AP_CONV[2]
    stage5_rgb = base_layers[7]
    stage5_lwir = base_layers[8]
    # stage6 = base_layers[9]
    stage6 = AP_CONV[3]
    stage6_rgb = base_layers[10]
    stage6_lwir = base_layers[11]

    P3_cls, P3_regr, P3_conf = IAFA(stage3,num_anchors[0], name='pred0_2_base', filters=filters, kersize=kersize, trainable=trainable)
    P4_cls, P4_regr, P4_conf = IAFA(stage4,num_anchors[1], name='pred1_2_base', filters=filters, kersize=kersize, trainable=trainable)
    P5_cls, P5_regr, P5_conf = IAFA(stage5,num_anchors[2], name='pred2_2_base', filters=filters, kersize=kersize, trainable=trainable)
    P6_cls, P6_regr, P6_conf = IAFA(stage6,num_anchors[3], name='pred3_2_base', filters=filters, kersize=kersize, trainable=trainable)

    y_cls = Concatenate(axis=1, name='mbox_cls_2')([P3_cls, P4_cls, P5_cls, P6_cls])
    y_regr = Concatenate(axis=1, name='mbox_regr_2')([P3_regr, P4_regr, P5_regr, P6_regr])
    conf_pred = Concatenate(axis=1, name='mbox_conf_2')([P3_conf, P4_conf, P5_conf, P6_conf])
    return [y_cls, y_regr], conf_pred


subtract_feature = Lambda(lambda x:tf.subtract(x[0],x[1]))


def create_AP_IAFA(base_layers, num_anchors,trainable=True):
    stage3 = base_layers[0]
    stage4 = base_layers[3]
    stage5 = base_layers[6]
    stage6 = base_layers[9]

    stage3 = Convolution2D(256, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage3_conv_new', trainable=trainable)(stage3)
    stage4 = Convolution2D(512, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage4_conv_new', trainable=trainable)(stage4)
    stage5 = Convolution2D(512, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage5_conv_new', trainable=trainable)(stage5)
    stage6 = Convolution2D(256, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage6_conv_new', trainable=trainable)(stage6)
    
    
    AP_CONV = [stage3,stage4,stage5,stage6]
    AP_predict = AP_stage(AP_CONV,base_layers, num_anchors, trainable=trainable)
    IAFA_predict, conf_pred = IAFA_stage(AP_CONV,base_layers, num_anchors, trainable=trainable)


    # additional branch
    stage3_rgb = base_layers[1]
    stage4_rgb = base_layers[4]
    stage5_rgb = base_layers[7]
    stage6_rgb = base_layers[10]

    stage3_ir = base_layers[2]
    stage4_ir = base_layers[5]
    stage5_ir = base_layers[8]
    stage6_ir = base_layers[11]


    stage3_rgb = Convolution2D(256, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage3_conv_rgb', trainable=trainable)(stage3_rgb)
    stage4_rgb = Convolution2D(512, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage4_conv_rgb', trainable=trainable)(stage4_rgb)
    stage5_rgb = Convolution2D(512, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage5_conv_rgb', trainable=trainable)(stage5_rgb)
    stage6_rgb = Convolution2D(256, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage6_conv_rgb', trainable=trainable)(stage6_rgb)
    
    stage3_ir = Convolution2D(256, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage3_conv_ir', trainable=trainable)(stage3_ir)
    stage4_ir = Convolution2D(512, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage4_conv_ir', trainable=trainable)(stage4_ir)
    stage5_ir = Convolution2D(512, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage5_conv_ir', trainable=trainable)(stage5_ir)
    stage6_ir = Convolution2D(256, (1,1), padding='same', activation='relu',
                      kernel_initializer='glorot_normal', name='stage6_conv_ir', trainable=trainable)(stage6_ir)

    AB_CONV = [stage3_rgb,stage4_rgb,stage5_rgb,stage6_rgb, stage3_ir, stage4_ir, stage5_ir, stage6_ir]
    AB_predict = AB_stage(AB_CONV,base_layers, num_anchors, trainable=trainable)
    AB_predict2, conf_addition = AB_stage2(AB_CONV,base_layers, num_anchors, trainable=trainable)

    return AP_predict, IAFA_predict, conf_pred, AB_predict, AB_predict2, conf_addition

