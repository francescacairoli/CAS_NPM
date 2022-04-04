import numpy as np
from numpy.random import rand
import scipy.special
import copy

class ICP_Classification():
	'''
	Inductive Conformal Prediction for a generic binary classification problem whose output is the probability of assigning 
	an input point to class 1

	Xc: input points of the calibration set
	Yc: labels corresponding to points in the calibration set
	mondrian_flag: if True computes class conditional p-values
	trained_model: function that takes x as input and returns the prob. of associating it to the positive class

	Remark: the default labels are 0 (negative class) and 1 (positive class)
	Careful: if different labels are considered, used the method set_labels
			(the non conformity scores are not well-defined otherwise)
	'''


	def __init__(self, dataset, trained_model, mondrian_flag):
		
		self.dataset = dataset
		self.mondrian_flag = mondrian_flag 
		self.trained_model = trained_model
		self.q = dataset.n_cal_points

		# n_classes, n_outputs
		

	def set_calibration_scores(self):

		self.cal_pred_lkh = self.trained_model(self.dataset.X_cal_scaled)
		self.calibr_scores = []
		for j in range(self.dataset.n_outputs):
			self.calibr_scores.append(self.get_nonconformity_scores(self.dataset.L_cal[:,j], self.cal_pred_lkh[:,:,j]))


	def get_nonconformity_scores(self, y_j, pred_lkh_j, sorting = True):

		pred_probs_j = scipy.special.softmax(pred_lkh_j, axis=1)
		n_points = len(y_j)
		ncm = np.array([np.abs(1-pred_probs_j[i,int(y_j[i])]) for i in range(n_points)])
		if sorting:
			ncm = np.sort(ncm)[::-1] # descending order
		return ncm


	def get_p_values(self, x):
		'''
		calibr_scores: non conformity measures computed on the calibration set and sorted in descending order
		x: new input points (shape: (n_points,x_dim)
		
		return: p-values for each class
		
		'''
		pred_lkh = self.trained_model(x) # prob of going to pos class on x
		n_points = len(pred_lkh)
		pvalues = []
		for j in range(self.dataset.n_outputs):
			p_values_j = np.empty((n_points, self.dataset.n_classes))
			if self.mondrian_flag:
				alphas_j = []
				for k in range(self.dataset.n_classes):
					alphas_j.append(self.calibr_scores[j][(self.calibr_scores[j] == k)])
				q_j = [alphas_j[ii].shape[0] for ii in range(self.dataset.n_classes)]
			else:
				alphas_j = [self.calibr_scores[j] for kk in range(self.dataset.n_classes)]
				q_j = [self.q for kk in range(self.dataset.n_classes)]

			A_j = []			
			for k in range(self.dataset.n_classes):
				A_j.append(self.get_nonconformity_scores(k*np.ones(n_points), pred_lkh[:,:,j], sorting = False)) # calibr scores for positive class
			
			

			for i in range(n_points):
				Ca = np.zeros(self.dataset.n_classes)
				Cb = np.zeros(self.dataset.n_classes)
				for k in range(self.dataset.n_classes):
					for qk in range(q_j[k]):
						if alphas_j[k][qk] > A_j[k][i]:
							Ca[k] += 1
						elif alphas_j[k][qk] == A_j[k][i]:
							Cb[k] += 1
						else:
							break
					
					p_values_j[i,k] = ( Ca[k] + rand() * (Cb[k] + 1) ) / (q_j[k] + 1)
			pvalues.append(p_values_j)
		return pvalues


	def get_confidence_credibility(self, pvalues):
		# INPUTS: p_pos and p_neg are the outputs returned by the function get_p_values
		# OUTPUT: array containing confidence and credibility [shape: (n_points,2)]
		# 		first column: confidence (1-smallest p-value)
		# 		second column: credibility (largest p-value)
		n_points = pvalues[0].shape[0]
		ConfCred = []
		for j in range(self.dataset.n_outputs):
			confidence_credibility_j = np.zeros((n_points,2))
			
			for i in range(n_points):
				Pji = pvalues[j][i]
				sort_indx_ji = np.argsort(Pji)
				confidence_credibility_j[i,0] = 1-Pji[sort_indx_ji[-2]] # confidence
				confidence_credibility_j[i,1] = Pji[sort_indx_ji[-1]] # credibility
			ConfCred.append(confidence_credibility_j)

		return ConfCred

	def compute_confidence_credibility(self, x):
		pvalues = self.get_p_values(x)

		return self.get_confidence_credibility(pvalues)


	def get_prediction_region(self, epsilon, pvalues):
		# INPUTS: p_pos and p_neg are the outputs returned by the function get_p_values
		#		epsilon = confidence_level
		# OUTPUT: one-hot encoding of the prediction region [shape: (n_points,2)]
		# 		first column: negative class
		# 		second column: positive class
		n_points = pvalues[0].shape[0]

		PredRegion = []
		for j in range(self.dataset.n_outputs):
			pred_region_j = np.zeros((n_points,self.dataset.n_classes)) 
			for i in range(n_points):
				for k in range(self.dataset.n_classes):
					if pvalues[j][i,k] > epsilon:
						pred_region_j[i,k] = 1
			PredRegion.append(pred_region_j)

		return PredRegion


	def get_coverage(self, pred_region, labels):

		n_points = len(labels)
		Cov = np.empty(self.dataset.n_outputs)
		
		for j in range(self.dataset.n_outputs):
			c = 0
			for i in range(n_points):
				if pred_region[j][i,int(labels[i,j])] == 1:
					c += 1

			Cov[j] = c/n_points

		return Cov

	def compute_coverage(self, eps, inputs, outputs):
		pvalues = self.get_p_values(x = inputs)

		self.pred_region = self.get_prediction_region(epsilon = eps, pvalues = pvalues)
		return self.get_coverage(self.pred_region, outputs)



	def  compute_cross_confidence_credibility(self):
		'''
		alphas: non conformity measures sorted in descending order
		cal_pred_lkhs: prediction likelihood fro the twoclasses on points of the calibration set 

		return: 2-dim array containing values of confidence and credibility (which are not exactly the p-values)
				shape = (n_points,2) -- CROSS VALIDATION STRATEGY
				first column: confidence (1-smallest p-value)
		# 		second column: credibility (largest p-value)
		'''
		pvalues = []
		for j in range(self.dataset.n_outputs):
			A_j = []
			for k in range(self.dataset.n_classes):
				A_j.append(self.get_nonconformity_scores(k*np.ones(self.q), self.cal_pred_lkh[:,:,j], sorting = False)) 
			
			pvalues_j = np.empty((self.q, self.dataset.n_classes))
			confidence_credibility_j = np.empty((self.q,2))

			
			for i in range(self.q):
				alphas_mask = copy.deepcopy(self.calibr_scores[j])
				alphas_mask = np.delete(alphas_mask,i)
				
				Ca = np.zeros(self.dataset.n_classes)
				Cb = np.zeros(self.dataset.n_classes)

				for k in range(self.dataset.n_classes):
					for qk in range(self.q-1):
						if alphas_mask[qk] > A_j[k][i]:
							Ca[k] += 1
						elif alphas_mask[qk] == A_j[k][i]:
							Cb[k] += 1
						else:
							break

					pvalues_j[i,k] = (Ca[k] + rand()*(Cb[k]+1) )/self.q
			

			pvalues.append(pvalues_j)

		self.cal_conf_cred = self.get_confidence_credibility(pvalues)		
			
		return self.cal_conf_cred


	def compute_efficiency(self):

		
		n_points = self.pred_region[0].shape[0]

		Eff = np.empty(self.dataset.n_outputs)

		for j in range(self.dataset.n_outputs):
			n_singletons = 0
			for i in range(n_points):
				if np.sum(self.pred_region[j][i]) == 1:
					n_singletons += 1
			Eff[j] = n_singletons/n_points
		return Eff
