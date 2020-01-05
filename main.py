from __future__ import division
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from layerClass import Layer
from dataClass import Data
from trainClass import train
from pathlib import Path
from freezeGraphClass import freezeGraph
from modelclass import Model
import numpy as np

filePath = Path("/data/schugh/experiment/newTrain/").glob('*.jpg')
jsonPath = Path("/data/schugh/experiment/newTrain/trainData.txt")
savePath = "/data/schugh/experiment/PupilIrisDenseNet/model/"
trainbatchSize = 16
valbatchSize = 16
epochs = 151
learningRate = 0.0001
outputTensorName =  "Inference/Output"
imageHeight = 240
imageWidth = 320
trainingLossList = list()
validationLossList = list()
data = Data(filePath, jsonPath)
data.jsonData()
trainData = data.loadLabels()
print ("train data shape is", trainData.shape)

def getnumberofBatches(Datasize, batchSize):
    return int(Datasize/batchSize)

with tf.Graph().as_default() as graph:
    with tf.Session() as sess:


        tf.constant([imageHeight,imageWidth], dtype="float32",name = "imageSize")
        tf.constant([outputTensorName], name = "OutputTensorName")
        X = tf.placeholder(tf.float32, [None, imageHeight,imageWidth,1], name='Input')
        Y = tf.placeholder(tf.float32, [None, imageHeight,imageWidth,2])
        model = Model(X,Y,learningRate)
        prediction = model.prediction()
        error = model.error()
        data.add_variable_summary(error, "Loss")
        optimizer = model.optimize()
        train_dataset, val_dataset, test_dataset = data.createTensorflowDatasets(0.8,0.1,0.1)
        merged_summary_operation = tf.summary.merge_all()
        modelname = "model_" + "learningRate_" + str(model.learningRate) + "epochs_" + str(epochs) 
        print (modelname) 
        train_summary_writer = tf.summary.FileWriter(savePath + modelname + '/tmp/' + modelname + "train")
        validation_summary_writer = tf.summary.FileWriter(savePath + modelname + '/tmp/' + modelname+ "validation")
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)
        print("Initialisation completed")
        trainingClass = train(sess,data,optimizer,error,model,merged_summary_operation)
        for epoch in range(epochs):
            print ("Current epoch is",epoch)
            batches = getnumberofBatches(data.train_size, trainbatchSize)
            print ("Number of batches", batches)
            trainingError = trainingClass.run(epoch, train_dataset,data.train_size, trainbatchSize,batches,train_summary_writer)
            print ("Training error is ", trainingError)
            trainingLossList.append(trainingError)
            batches = getnumberofBatches(data.val_size, valbatchSize)
            validationError = trainingClass.validation(epoch,val_dataset,data.val_size, valbatchSize ,batches,validation_summary_writer)
            print ("Validation Error is ", validationError)
            validationLossList.append(validationError)
            if (epoch % 10 == 0):
                saver = tf.train.Saver()
                save_path = saver.save(sess, savePath + modelname + "/" + str(epoch) + "/model.ckpt")
                print("Model saved in path: %s" % save_path)
                plt.figure()
                matplotlib.rcParams.update({'font.size':12})
                plt.plot(trainingLossList, 'r')
                plt.plot(validationLossList, 'b')
                plt.xlabel("Number of iterations")
                plt.ylabel("Error")
                plt.gca().legend(('training Loss','validation Loss'))
                plt.savefig(savePath + modelname + "/" + str(epoch) + '/Loss' + '.png')
                freezeGraph = freezeGraph(savePath + modelname + "/" + str(epoch) + "/", outputTensorName)
                freezeGraph.freeze_graph()  
        saver = tf.train.Saver()
        save_path = saver.save(sess, savePath + modelname + "/" + "model.ckpt")
        print("Model saved in path: %s" % save_path)

freezeGraph = freezeGraph(savePath + modelname + "/" + str(epoch) + "/", outputTensorName)
freezeGraph.freeze_graph()  
plt.figure()
matplotlib.rcParams.update({'font.size':12})
plt.plot(trainingLossList, 'r')
plt.plot(validationLossList, 'b')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.gca().legend(('Training Loss','Validation Loss'))
plt.savefig(savePath + modelname + "/" + str(epoch) + "/" + "CornealReflectionsLoss" + ".png")