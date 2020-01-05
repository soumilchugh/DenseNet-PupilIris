import cv2
import numpy as np
import json
import imutils
import glob
import os
from collections import defaultdict
import tensorflow as tf
from pathlib import Path

class Data(object):

	def __init__(self,filePath,jsonPath):
		self.filePath = filePath
		self.jsonPath = jsonPath
		self.trainPath = list()
		self.Dictdata = defaultdict(dict)
		self.train_dataset = None
		self.test_dataset = None
		self.val_dataset = None
		self.train_size = None
		self.val_size = None
		self.test_size = None

	def jsonData(self):
		with open(str(self.jsonPath), 'r') as f:
			self.Dictdata =  json.loads(f.read())

	def loadLabels(self):
		files = [file for file in self.filePath]
		for file in files:
			filename, file_extension = os.path.splitext(str(file))
			name = str(os.path.basename(filename));
			my_path = file.absolute().as_posix()
			if (name + ".jpg") in self.Dictdata:
				if "Dan" not in name:
					self.trainPath.append(str(my_path))
		return np.array(self.trainPath)

	def createTensorflowDatasets(self,trainSize, validationSize, testSize):
		PathDataset = tf.data.Dataset.from_tensor_slices(self.trainPath)
		self.trainPath = np.array(self.trainPath)
		#fullDataset = tf.data.Dataset.zip((PathDataset, PathDataset))
		self.train_size = int(trainSize*self.trainPath.shape[0])
		self.val_size = int(validationSize*self.trainPath.shape[0])
		self.test_size  = int(testSize*self.trainPath.shape[0])
		PathDataset.shuffle(self.trainPath.shape[0])
		self.train_dataset = PathDataset.take(self.train_size)
		test_dataset = PathDataset.skip(self.train_size)
		self.val_dataset = test_dataset.skip(self.val_size)
		self.test_dataset = test_dataset.take(self.test_size)
		return self.train_dataset, self.val_dataset, self.test_dataset

	def createDatasetIterator(self,dataset, datasetSize, batchSize):
		dataset = dataset.shuffle(datasetSize).batch(batchSize)
		datasetIterator = dataset.make_initializable_iterator()
		return datasetIterator

	def getBatchData(self, batch):
		finalData = list()
		for image in (batch):
			image_reader = cv2.imread(image.decode("utf-8"),0)
			normalised_image = image_reader.astype(np.float)/255.0
			normalised_image = np.expand_dims(normalised_image, axis=2)
			finalData.append(normalised_image)
		return finalData

	def getBatchLabels(self,batch):
		finalData = list()
		for image in (batch):
			filename, file_extension = os.path.splitext(image.decode("utf-8"))
			name = str(os.path.basename(filename));
			boundaryPoints = Dictdata[name + ".jpg"]['BoundaryPoints']
			irisBoundaryPoints = Dictdata[name + ".jpg"]['IrisBoundaryPoints']
			newBoundaryPoints = []
			newIrisBoundaryPoints = []
			for item in boundaryPoints:
				newBoundaryPoints.append((int(item[0]*320),int(item[1]*240)))

			labelList = list()
			mask = np.zeros((240, 320))

			if len(newBoundaryPoints) > 0:
				e = cv2.fitEllipse(np.array(newBoundaryPoints))
				newcenterX = int(e[0][0])
				newcenterY = int(e[0][1])
				major_x = int((e[1][0])/2)
				major_y = int((e[1][1])/2)
				angle = int(e[2])
				cv2.ellipse(mask, (newcenterX,newcenterY), (major_x, major_y), angle, 0, 360, (255,255,255), -1)
				mask = mask.astype(np.float)/255.0
				
			mask = np.expand_dims(mask, axis=2)
			labelList.append(mask)

			for item in irisBoundaryPoints:
				newIrisBoundaryPoints.append((int(item[0]*320),int(item[1]*240)))

			mask = np.zeros((240, 320))
			if len(newIrisBoundaryPoints) > 0:
				e = cv2.fitEllipse(np.array(newIrisBoundaryPoints))
				newcenterX = int(e[0][0])
				newcenterY = int(e[0][1])
				major_x = int((e[1][0])/2)
				major_y = int((e[1][1])/2)
				angle = int(e[2])
				cv2.ellipse(mask, (newcenterX,newcenterY), (major_x, major_y), angle, 0, 360, (255,255,255), -1)
				mask = mask.astype(np.float)/255.0
				mask = np.expand_dims(mask, axis=2)
			
			labelList.append(mask)

			finalData.append(np.concatenate(labelList,axis=2))
		return finalData

	def add_variable_summary(self,tf_variable, summary_name):
		with tf.name_scope(summary_name + '_summary'):
			mean = tf.reduce_mean(tf_variable)
			tf.summary.scalar('Mean',mean)
			with tf.name_scope('standard_deviation'):
				standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))
			tf.summary.scalar('StandardDeviation',standard_deviation)
			tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
			tf.summary.scalar('Minimum',tf.reduce_min(tf_variable))
			tf.summary.histogram('Histogram',tf_variable)



