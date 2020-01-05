import cv2
import numpy as np
import json
import imutils
import glob
import os
from collections import defaultdict
import tensorflow as tf
from pathlib import Path
from dataClass import Data

class train(Data):
	def __init__(self,sess,data, optimizer, error, model,merged_summary_operation):
		self.sess = sess
		self.data = data
		self.optimizer = optimizer
		self.error = error
		self.model = model
		self.merged_summary_operation = merged_summary_operation

	def run(self, epoch, dataset,size, sizeOfBatch,numberOfBatches,summary_writer):
		trainingErrorList = list()
		datasetIterator = self.data.createDatasetIterator(dataset,size, sizeOfBatch)
		batch = datasetIterator.get_next()
		self.sess.run(datasetIterator.initializer)
		for i in range(numberOfBatches):
			batch_data = self.sess.run(batch)
			_, loss = self.sess.run([self.optimizer,self.error], feed_dict={self.model.input:self.data.getBatchData(batch_data),self.model.output:self.data.getBatchLabels(batch_data)})
			trainingErrorList.append(loss)
			merged_summary = self.sess.run(self.merged_summary_operation,feed_dict={self.model.input:self.data.getBatchData(batch_data),self.model.output:self.data.getBatchLabels(batch_data)})
			summary_writer.add_summary(merged_summary,epoch)
			print (loss)
		return np.average(trainingErrorList)

	def validation(self, epoch, dataset,size, sizeOfBatch,numberOfBatches,summary_writer):
		trainingErrorList = list()
		datasetIterator = self.data.createDatasetIterator(dataset,size, sizeOfBatch)
		batch = datasetIterator.get_next()
		self.sess.run(datasetIterator.initializer)
		for i in range(numberOfBatches):
			batch_data = self.sess.run(batch)
			self.data.getBatchData(batch_data)
			loss = self.sess.run(self.error, feed_dict={self.model.input:self.data.getBatchData(batch_data),self.model.output:self.data.getBatchLabels(batch_data)})
			trainingErrorList.append((loss))
			merged_summary = self.sess.run(self.merged_summary_operation,feed_dict={self.model.input:self.data.getBatchData(batch_data),self.model.output:self.data.getBatchLabels(batch_data)})
			summary_writer.add_summary(merged_summary,epoch)
		return np.average(trainingErrorList)
