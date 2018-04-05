#!/usr/bin/env python2
# This Python file uses the following encoding: utf-8

#Defining the Encoding,To define a source code encoding, a magic comment must be placed into the source files either as first or second line in the file
#

# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time

start = time.time()

import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2) #浮点输出的精度位数

import openface

# fileDir = os.path.dirname(os.path.realpath(__file__))#获得你刚才所引用的模块 所在的绝对路径
# print(fileDir);
fileDir='/home/angelo/openface/demos';
# modelDir = os.path.join(fileDir, '..', 'models')#Join one or more path components intelligently,
# print(modelDir);
modelDir='/home/angelo/openface/demos/../models'
# dlibModelDir = os.path.join(modelDir, 'dlib')
# print(dlibModelDir);
dlibModelDir='/home/angelo/openface/demos/../models/dlib'
# openfaceModelDir = os.path.join(modelDir, 'openface')
# print(openfaceModelDir);
openfaceModelDir='/home/angelo/openface/demos/../models/openface'

# parser = argparse.ArgumentParser()
#
# parser.add_argument('imgs', type=str, nargs='+', help="Input images.",
#                     default=['demos/adams.jpg','demos/carell.jpg'])
# parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
#                     default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
# parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
#                     default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
# parser.add_argument('--imgDim', type=int,
#                     help="Default image dimension.", default=96)
# parser.add_argument('--verbose', action='store_true')

args = argparse.Namespace(dlibFacePredictor='/home/angelo/openface/demos/../models/dlib/shape_predictor_68_face_landmarks.dat', imgDim=96, imgs=['adams.jpg', 'carell.jpg'], networkModel='/home/angelo/openface/demos/../models/openface/nn4.small2.v1.t7', verbose=False)

# print(args)
# print(args.imgs)
# print(args.dlibFacePredictor)
# print(args.networkModel)
# print(args.imgDim)
# print(args.verbose)
if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
print(start);
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    print(imgPath);
    bgrImg = cv2.imread(imgPath)# read an image
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)# create a grayscale version

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)#找到图像中最大的人脸边界框。
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)#变换并对齐图像中的人脸。
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)#执行RGB图像的前向网络传递。
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return rep

if __name__=="__main__":
    for (img1, img2) in itertools.combinations(args.imgs, 2):
        d = getRep(img1) - getRep(img2)
        print("Comparing {} with {}.".format(img1, img2))
        print(
            "  + Squared l2 distance between representations: {:0.3f}".format(np.dot(d, d)))
