import pickle
import numpy as np

class Dataset():

	def __init__(self, trainset_fn, testset_fn, po_flag = False):
		self.trainset_fn = trainset_fn
		self.testset_fn = testset_fn
		self.po_flag = po_flag

	def add_calibration_path(self, calibrset_fn):
		self.calibrset_fn = calibrset_fn

	def load_data(self):
		self.load_train_data()
		self.load_test_data()
		
	def load_train_data(self,):

		file = open(self.trainset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		self.T_train = np.transpose(data["labels"],(0,2,1))
		self.L_train = np.argmax(self.T_train, axis = 1)
		print(self.T_train)

		self.n_stations = self.T_train.shape[1]
		self.station_idxs = np.eye(self.n_stations).flatten()
		if self.po_flag:	
			self.X_train = data["x0"][:,self.station_idxs]
		else:
			self.X_train = data["x0"]

		self.n_outputs = self.T_train.shape[2]
		self.n_classes = self.T_train.shape[1]
		self.input_dim = self.X_train.shape[1]
		self.n_training_points = self.X_train.shape[0]
		
		self.xmax = np.max(self.X_train, axis = 0)
		self.xmin = np.min(self.X_train, axis = 0)

		self.X_train_scaled = -1+2*(self.X_train-self.xmin)/(self.xmax-self.xmin)
		
		
		
	def load_test_data(self):

		file = open(self.testset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		self.T_test = np.transpose(data["labels"],(0,2,1))
		self.L_test = np.argmax(self.T_test, axis = 1)
		
		if self.po_flag:	
			self.X_test = data["x0"][:,self.station_idxs]
		else:
			self.X_test = data["x0"]

		self.n_test_points = self.X_test.shape[0]
		
		self.X_test_scaled = -1+2*(self.X_test-self.xmin)/(self.xmax-self.xmin)


	def load_calibration_data(self):

		file = open(self.calibrset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		self.T_cal = np.transpose(data["labels"],(0,2,1))
		self.L_cal = np.argmax(self.T_cal, axis = 1)
		
		if self.po_flag:	
			self.X_cal = data["x0"][:,self.station_idxs]
		else:
			self.X_cal = data["x0"]

		self.n_cal_points = self.X_cal.shape[0]
		
		self.X_cal_scaled = -1+2*(self.X_cal-self.xmin)/(self.xmax-self.xmin)

		
	def generate_mini_batches(self, n_samples):
		
		ix = np.random.randint(0, self.n_training_points, n_samples)
		Xb = self.X_train_scaled[ix]
		Tb = self.T_train[ix] # one hot encoding
		Lb = self.L_train[ix] # label = class index
		
		return Xb, Tb, Lb

