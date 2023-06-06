# -*- coding:UTF-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras.layers import *
from keras import backend as K
from .FixedBatchNormalization import FixedBatchNormalization
import numpy as np
import tensorflow as tf
from keras.layers import Multiply, multiply
from .keras_layer_L2Normalization import L2Normalization
from .scale_bias import Scale_bias


def identity_block(input_tensor, input_tensor_mix,kernel_size, filters, stage, block, dila=(1, 1), modality ='',noadd = True,trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a'+modality, trainable=trainable)(input_tensor_mix)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a'+modality)(x)
    x = Activation('relu')(x)
    #x = Dropout(0.2)(x, training=True)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=dila, padding='same',
                      name=conv_name_base + '2b'+modality, trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b'+modality)(x)
    x = Activation('relu')(x)
    #x = Dropout(0.2)(x, training=True)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c'+modality, trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c'+modality)(x)
    #add in the DMAF block
    if noadd:
        return x,input_tensor
    #normal
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    #x = Dropout(0.2)(x, training=True)
    return x


def conv_block(input_tensor, input_tensor_mix,kernel_size, filters, stage, block, strides=(2, 2), dila=(1, 1), modality ='',noadd = True,trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a'+modality, trainable=trainable)(
        input_tensor_mix)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a'+modality)(x)
    x = Activation('relu')(x)
    #x = Dropout(0.2)(x, training=True)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=dila, padding='same',
                      name=conv_name_base + '2b'+modality, trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b'+modality)(x)
    x = Activation('relu')(x)
    #x = Dropout(0.2)(x, training=True)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c'+modality, trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c'+modality)(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1'+modality, trainable=trainable)(
        input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1'+modality)(shortcut)
    #add in the DMAF block
    if noadd:
        return x,shortcut
    #normal
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    #x = Dropout(0.2)(x, training=True)
    return x


weight_average = Lambda(lambda x: x * 0.5)
weight_multiply = Lambda(lambda x: x[0] * x[1])
weight_div = Lambda(lambda x: tf.div(x[0] , x[1]))
subtract_feature = Lambda(lambda x:tf.subtract(x[0],x[1]))



def fusion(x, x_lwir, upblock=None, trainable=False, add=True, stage=2, block=None, channel_size=64):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_axis = 3

    x_fusion = Concatenate()([x, x_lwir])
    x_fusion = Convolution2D(channel_size*4, (1, 1), strides=(1, 1), name=conv_name_base + '_fusion', padding='same',
                          kernel_initializer='glorot_normal', trainable=trainable)(x_fusion)
    x_fusion = FixedBatchNormalization(axis=bn_axis, name=conv_name_base+'_fusion_bn_new')(x_fusion)
    if add==False:
        x_fusion = Activation('relu')(x_fusion)
        #x_fusion = Dropout(0.2)(x_fusion, training=True)
        return x_fusion

    if block == 'a':
        upblock = Convolution2D(channel_size*4, (1, 1), strides=(2, 2), name=conv_name_base + '_fusion2', padding='same',
                          kernel_initializer='glorot_normal', trainable=trainable)(upblock)
        upblock = FixedBatchNormalization(axis=bn_axis, name=conv_name_base+'_upblock_bn_new')(upblock)
    x_fusion = Add()([x_fusion, upblock])
    x_fusion = Activation('relu')(x_fusion)
    #x_fusion = Dropout(0.2)(x_fusion, training=True)
    return x_fusion


def ResNet_Block(x,x_lwir,x_mix,x_lwir_mix, stage_order =3,identity_block_num = 0,channel_size = 128, trainable = False, fusion_map = None):
    block_order=['a','b','c','d','e','f']

    x, x_shortcut = conv_block(x, x_mix, 3, [channel_size, channel_size, channel_size*4], stage=stage_order, block='a', strides=(2, 2),
                               modality='', noadd=True, trainable=trainable)
    x_lwir, x_lwir_shortcut = conv_block(x_lwir, x_lwir_mix, 3, [channel_size, channel_size, channel_size*4], stage=stage_order, block='a',
                                         strides=(2, 2), modality='_lwir', noadd=True, trainable=trainable)
    x_pre = Add()([x, x_shortcut])
    x_lwir_pre = Add()([x_lwir, x_lwir_shortcut])
    x = Activation('relu')(x_pre)
    #x = Dropout(0.1)(x, training=True)
    x_lwir = Activation('relu')(x_lwir_pre)
    #x_lwir = Dropout(0.1)(x_lwir, training=True)
    if identity_block_num==0:
        fusion_map = fusion(x, x_lwir, fusion_map, trainable=trainable, stage=stage_order, block='a', channel_size=channel_size)
        return x, x_lwir, fusion_map
    x_mix, x_lwir_mix = x, x_lwir
    fusion_map = fusion(x_mix, x_lwir_mix, fusion_map, trainable=trainable, stage=stage_order, block='a', channel_size=channel_size)

    for identity_i in range(identity_block_num):
        x,x_input_tensor = identity_block(x,x_mix, 3, [channel_size, channel_size, channel_size*4], stage=stage_order, block=block_order[identity_i+1], modality='',noadd = True,trainable = trainable)
        x_lwir,x_lwir_input_tensor = identity_block(x_lwir, x_lwir_mix,3, [channel_size, channel_size, channel_size*4],stage=stage_order, block=block_order[identity_i+1],modality='_lwir',noadd = True, trainable=trainable)
        x_pre = Add()([x, x_input_tensor])
        x_lwir_pre = Add()([x_lwir,x_lwir_input_tensor])
        x = Activation('relu')(x_pre)
        # x = Dropout(0.05)(x, training=True)
        x_lwir = Activation('relu')(x_lwir_pre)
        # x_lwir = Dropout(0.05)(x_lwir, training=True)
        x_mix, x_lwir_mix = x, x_lwir
        fusion_map = fusion(x_mix, x_lwir_mix, fusion_map, trainable=trainable, stage=stage_order, block=block_order[identity_i+1], channel_size=channel_size)

    return  x,x_lwir,x_mix,x_lwir_mix, fusion_map



def Backbone(input_tensor_rgb=None, input_tensor_lwir=None,trainable=False):
    img_input_rgb = input_tensor_rgb
    img_input_lwir = input_tensor_lwir
    if K.image_dim_ordering() == 'tf':#bn_axis = 3
        bn_axis = 3
    else:
        bn_axis = 1
  
    print('Froze the first two stage layers')
    x = ZeroPadding2D((3, 3))(img_input_rgb)
    x_lwir = ZeroPadding2D((3, 3))(img_input_lwir)
    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable = False)(x)
    x_lwir = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1_lwir', trainable=False)(x_lwir)
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x_lwir = FixedBatchNormalization(axis=bn_axis, name='bn_conv1_lwir')(x_lwir)
    x = Activation('relu')(x)
    x_lwir = Activation('relu')(x_lwir)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x_lwir = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x_lwir)
    x = conv_block(x, x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),modality='', noadd = False,trainable = False)
    x_lwir = conv_block(x_lwir, x_lwir, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), modality='_lwir',noadd = False,trainable=False)
    x = identity_block(x, x, 3, [64, 64, 256], stage=2, block='b', modality='', noadd = False,trainable =False )
    x_lwir = identity_block(x_lwir, x_lwir, 3, [64, 64, 256], stage=2, block='b',  modality='_lwir',noadd = False,trainable=False)
    stage2_rgb = identity_block(x, x, 3, [64, 64, 256], stage=2, block='c',modality='',noadd = False,  trainable = False)
    stage2_lwir = identity_block(x_lwir, x_lwir, 3, [64, 64, 256], stage=2, block='c',  modality='_lwir',noadd = False,trainable=False)

    print('the backbone (ResNet50)')
    fusion_map = fusion(stage2_rgb, stage2_lwir, add=False, trainable=trainable, stage=2, block='a')
    stage3_rgb,stage3_lwir,stage3_rgb_mix,stage3_lwir_mix, fusion_map3 = ResNet_Block(stage2_rgb, stage2_lwir, stage2_rgb, stage2_lwir,stage_order=3, identity_block_num=3, channel_size=128, trainable=trainable, fusion_map=fusion_map)
    stage4_rgb,stage4_lwir,stage4_rgb_mix,stage4_lwir_mix, fusion_map4 = ResNet_Block(stage3_rgb, stage3_lwir,stage3_rgb_mix,stage3_lwir_mix,stage_order=4,identity_block_num=5, channel_size=256,trainable=trainable, fusion_map=fusion_map3)
    stage5_rgb,stage5_lwir,stage5_rgb_mix,stage5_lwir_mix, fusion_map5 = ResNet_Block(stage4_rgb,stage4_lwir,stage4_rgb_mix,stage4_lwir_mix,stage_order=5,identity_block_num=2, channel_size=512,trainable=trainable, fusion_map=fusion_map4)
    stage6_rgb,stage6_lwir, fusion_map6 = ResNet_Block(stage5_rgb,stage5_lwir,stage5_rgb_mix,stage5_lwir_mix,stage_order=6,identity_block_num=0, channel_size=256,trainable=trainable, fusion_map=fusion_map5)
   
    stage3, stage4, stage5, stage6 = fusion_map3, fusion_map4, fusion_map5, fusion_map6
    
    predictor_sizes = np.array([stage3._keras_shape[1:3],
                                stage4._keras_shape[1:3],
                                stage5._keras_shape[1:3],
                                np.ceil(np.array(stage5._keras_shape[1:3]) / 2)])

    return [stage3,stage3_rgb,stage3_lwir,  stage4,stage4_rgb,stage4_lwir,
            stage5,stage5_rgb,stage5_lwir,  stage6,stage6_rgb,stage6_lwir],\
           predictor_sizes

