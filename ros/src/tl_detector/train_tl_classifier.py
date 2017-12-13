#!/usr/bin/env python
#import subprocess
#subprocess.call("source /home/mikkel/udacity/Term3/CarND-Capstone_personal/ros/devel/setup.bash", shell=True)
from __future__ import division, print_function, absolute_import
import rospy
from std_msgs.msg import Int32
#from geometry_msgs.msg import PoseStamped, Pose
#from styx_msgs.msg import TrafficLightArray, TrafficLight
#from styx_msgs.msg import Lane
#from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
#import tf
import tensorflow as tf
import cv2
import yaml
import numpy as np
import os

import glob
import re
import shutil
#import pandas

PREPARE_TRAINING_SET = False
PREPARE_TRAINING_SET_EQUAL_SIZE = False
PREPARE_TEST_SET = False
PREPARE_TEST_SET_EQUAL_SIZE = False
TRAIN_NETWORK = False
TEST_NETWORK = False

classes_text = ['red','yellow','green']
use_classes = [0,1,2] # Only train on these classes (e.g. [0,2] for only training red/green distinction)

src_dir = '../../../data/traffic_light_images_simulator/'

CLASSIFIER = 'light_classification/models/1513151224/' # Trained model

#### Do not change below ####
classes = range(len(classes_text))

src_dir_train = os.path.join(src_dir,'train')
src_dir_test = os.path.join(src_dir,'test')

cropped_dir_train = os.path.join(src_dir_train,'cropped_rcnn')
equal_size_dir_train = os.path.join(src_dir_train,'equal_size_rcnn')
cropped_dir_test = os.path.join(src_dir_test,'cropped_rcnn')
equal_size_dir_test = os.path.join(src_dir_test,'equal_size_rcnn')

light_classifier = TLClassifier()

if PREPARE_TRAINING_SET:
    if os.path.exists(cropped_dir_train):
        shutil.rmtree(cropped_dir_train)
    os.makedirs(cropped_dir_train)    
    
    light_classifier = TLClassifier()
    files = glob.glob(os.path.join(src_dir_train,'raw','*.jpg'))
    #for sFile in files:
    for s in range(len(files)):
        print(str(s) + '/' + str(len(files)-1))
        sFile = files[s]
        img = cv2.imread(sFile)
        image_np, box_coords, classes, scores = light_classifier.detect_tl(img)
        
        # Run through bounding boxes with detected traffic lights and save each as an image
        for i in range(len(box_coords)):
            bot, left, top, right = box_coords[i, ...]
            class_id = int(classes[i])
            score = scores[i]
            h = top - bot
            w = right - left
            if h <= 1.5 * w:
                continue # Truncated Traffic Ligth box

            tl_img = image_np[int(bot):int(top), int(left):int(right)]
            
            dstFile = os.path.join(cropped_dir_train,sFile.split('/')[-1][:-4] + '_' + str(i) + '.png')
            cv2.imwrite(dstFile, tl_img)

if PREPARE_TRAINING_SET_EQUAL_SIZE:
    if os.path.exists(equal_size_dir_train):
        shutil.rmtree(equal_size_dir_train)
    os.makedirs(equal_size_dir_train)    

    files = glob.glob(os.path.join(cropped_dir_train,"*.png"))
    
    for s in range(len(files)):
        print(str(s) + '/' + str(len(files)-1))
        sFile = files[s]
        img = cv2.imread(sFile)
        equal_size_img = cv2.resize(img,dsize=(30,90))
        dstFile = os.path.join(equal_size_dir_train,sFile.split('/')[-1])
        cv2.imwrite(dstFile, equal_size_img)
        
if PREPARE_TEST_SET:
    if os.path.exists(cropped_dir_test):
        shutil.rmtree(cropped_dir_test)
    os.makedirs(cropped_dir_test)    
    
    light_classifier = TLClassifier()

    for c in range(len(classes_text)):
        class_text = classes_text[c]
        files = glob.glob(os.path.join(src_dir_test,'raw',class_text,"*.jpg"))

        for s in range(len(files)):
            print(str(s) + '/' + str(len(files)-1))
            sFile = files[s]
            img = cv2.imread(sFile)
            image_np, box_coords, classes, scores = light_classifier.detect_tl(img)
            
            # Run through bounding boxes with detected traffic lights and save each as an image
            for i in range(len(box_coords)):
                bot, left, top, right = box_coords[i, ...]
                class_id = int(classes[i])
                score = scores[i]
                h = top - bot
                w = right - left
                if h <= 1.5 * w:
                    continue # Truncated Traffic Ligth box

                tl_img = image_np[int(bot):int(top), int(left):int(right)]
                
                dstFile = os.path.join(cropped_dir_test,sFile.split('/')[-1][:-4] + '_' + str(c) + '_' + str(i) + '.png')
                cv2.imwrite(dstFile, tl_img)

if PREPARE_TEST_SET_EQUAL_SIZE:
    if os.path.exists(equal_size_dir_test):
        shutil.rmtree(equal_size_dir_test)
    os.makedirs(equal_size_dir_test)    

    files = glob.glob(os.path.join(cropped_dir_test,"*.png"))    
    
    #for sFile in files:
    for s in range(len(files)):
        print(str(s) + '/' + str(len(files)-1))
        sFile = files[s]
        img = cv2.imread(sFile)
        equal_size_img = cv2.resize(img,dsize=(30,90))
        dstFile = os.path.join(equal_size_dir_test,sFile.split('/')[-1])
        cv2.imwrite(dstFile, equal_size_img)

if TRAIN_NETWORK:
    import tensorflow as tf
    
    # Training Parameters
    learning_rate = 0.0001
    num_steps = 300
    batch_size = 16
    
    # Network Parameters
    num_classes = len(use_classes)
    dropout = 0.75 # Dropout, probability to keep units
    
    # Prepare training data
    label_dist_list = []
    image_list = [] 
    images_classes = [[] for i in range(len(classes))]
    files = glob.glob(os.path.join(equal_size_dir_train,"*.png"))
    for s in range(len(files)):
        sFile = files[s]
        img = cv2.imread(sFile)
        str_parts = re.split('[/ .]',sFile)[-2]
        label = int(str_parts.split('_')[1])
        images_classes[label].append(img.reshape(-1))
        
    images_classes = [np.asarray(x) for x in images_classes]        
    n_samples = min([len(x) for x in images_classes])
    
    images = []
    labels = []
    for c in range(len(use_classes)):
        rand_class_inds = np.random.permutation(n_samples)
        for s in range(len(rand_class_inds)):
            images.append(images_classes[use_classes[c]][rand_class_inds[s]])
            labels.append(c)
            
    # Shuffle data (and classes!)
    rand_inds = np.random.permutation(len(images))
    images = [images[i] for i in rand_inds]
    labels = [labels[i] for i in rand_inds]
    
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels)
    
    # Prepare test data
    images_test = []
    labels_test = []
    files = glob.glob(os.path.join(equal_size_dir_test,"*.png"))
    for s in range(len(files)):
        sFile = files[s]
        img = cv2.imread(sFile)
        str_parts = re.split('[/ .]',sFile)[-2]
        label = int(str_parts.split('_')[1])
        if np.isin([label],use_classes)[0]:
            labels_test.append(use_classes.index(label))
            images_test.append(img.reshape(-1))
    
    images_test = np.asarray(images_test, dtype=np.float32)
    labels_test = np.asarray(labels_test)

    # Inspiration from: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
    # Create the neural network
    def conv_net(x_dict, n_classes, dropout, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs
            x = x_dict['x']
            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(x, shape=[-1, 90, 30, 3])
            
            # Normalize data per image (to better generalize to unseen data distributions)
            x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)
            
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x, 16, 3, activation=tf.nn.relu, kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
            
            # Convolution Layer with 32 filters and a kernel size of 5
            conv2 = tf.layers.conv2d(conv1, 32, 3, activation=tf.nn.relu, kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    
            # Convolution Layer with 64 filters and a kernel size of 3
            conv3 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu, kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
    
            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv3)
    
            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, 1024, kernel_initializer= tf.random_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
    
            # Output layer, class prediction
            out = tf.layers.dense(fc1, n_classes)
        return out
    
    # Define the model function (following TF Estimator Template)
    def model_fn(features, labels, mode):
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = conv_net(features, num_classes, dropout, reuse=False,
                                is_training=True)
        logits_test = conv_net(features, num_classes, dropout, reuse=True,
                               is_training=False)
    
        # Predictions
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_probas = tf.nn.softmax(logits_test)
    
        # If prediction mode, early return
        LABELS = [classes_text[c] for c in use_classes]
        label_values = tf.constant(LABELS)
        outputs = tf.gather(label_values,pred_classes)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes,
            export_outputs={'class': tf.estimator.export.ClassificationOutput(classes=outputs)})
    
        # Define loss and optimizer
        cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)   
        loss_op = cross_entropy_loss+sum(reg_losses)
            
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())
    
        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    
        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op},
            export_outputs={'class': tf.estimator.export.ClassificationOutput(classes=outputs)})
    
        return estim_specs
    
    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)
    
    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': images}, y=labels,
        batch_size=batch_size, num_epochs=None, shuffle=True)    
    
    # Train the Model
    model.train(input_fn, steps=num_steps)
    
    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': images_test}, y=labels_test,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)
    
    print("Testing Accuracy:", e['accuracy'])   
    
    # Save the model
    def serving_input_receiver_fn():
        feature_spec = {'x': tf.FixedLenSequenceFeature([30,90,3],dtype=tf.float32,allow_missing=True)}
        serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')
        receiver_tensors = {'inputs': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    
    CLASSIFIER = model.export_savedmodel(export_dir_base='light_classification/models',serving_input_receiver_fn=serving_input_receiver_fn)

if TEST_NETWORK:
    def _float32_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    images_test = []
    labels_test = []
    files = glob.glob(os.path.join(equal_size_dir_test,'*.png'))
    for s in range(len(files)):
        sFile = files[s]
        img = cv2.imread(sFile)
        str_parts = re.split('[/ .]',sFile)[-2]
        label = int(str_parts.split('_')[1])
        if np.isin([label],use_classes)[0]:
            labels_test.append(use_classes.index(label))
            images_test.append(img)
    
    images_test = np.asarray(images_test, dtype=np.float32)    
    
    with tf.Session(graph=tf.Graph()) as sess:
        # Load model
        graph_file = os.path.join(CLASSIFIER,'saved_model.pb')
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], CLASSIFIER)        
        predictor = tf.contrib.predictor.from_saved_model(CLASSIFIER)
        
        # Prepare input examples
        input_tensor=tf.get_default_graph().get_tensor_by_name('input_tensors:0')
        model_input= tf.train.Example(features=tf.train.Features(feature={'x': _float32_feature(images_test[0:100].reshape(-1))})) 
        model_input=model_input.SerializeToString()
        
        # Perform inference (classification)
        output_dict= predictor({"inputs":[model_input]})
        
        # Print
        print(" prediction is " , output_dict['classes'])
        print(" ground truth is ", [classes_text[c] for c in labels_test[0:100]])