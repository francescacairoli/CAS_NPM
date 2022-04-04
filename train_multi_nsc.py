from MultiNSC import *
import numpy as np
import os
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

cuda = True if torch.cuda.is_available() else False
#cuda = False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class Train_MultiNSC():
	def __init__(self, model_name, arch_name, dataset, training_flag = True, idx = None):
		
		self.model_name = model_name
		self.arch_name = arch_name
		self.dataset = dataset
		self.idx = idx
		self.training_flag = training_flag
		if self.idx:
			self.results_path = self.arch_name+"/"+self.model_name+"/ID_"+self.idx
		
	
	def compute_accuracy(self, real_label, hypothesis):
		
		n_points = real_label.size(0)

		global_correct_preds = 0
		peroutput_correct_preds = np.zeros(self.dataset.n_outputs)
		for i in range(n_points):
			for j in range(self.dataset.n_outputs):
				pred_label_ij = hypothesis[i,:,j].data.argmax()
				if pred_label_ij == real_label[i,j]:
					global_correct_preds += 1
					peroutput_correct_preds[j] += 1

		global_accuracy = global_correct_preds/(n_points*self.dataset.n_outputs)
		peroutput_accuracy = peroutput_correct_preds/n_points 

		return global_accuracy, peroutput_accuracy


	def train(self, n_epochs, batch_size, lr):

		self.idx = str(np.random.randint(0,100000))
		print("ID = ", self.idx)

		self.results_path = self.arch_name+"/"+self.model_name+"/ID_"+self.idx
		os.makedirs(self.results_path, exist_ok=True)

		self.dataset.load_data()
		self.n_epochs = n_epochs
		
		self.mnsc = MultiNSC(input_size = int(self.dataset.input_dim), hidden_size = 20, output_size = self.dataset.n_outputs*self.dataset.n_classes)
		
		if cuda:
			self.mnsc.cuda()

		loss_fnc = nn.CrossEntropyLoss()    # Softmax is internally computed.

		optimizer = torch.optim.Adam(self.mnsc.parameters(), lr=lr)

		self.net_path = self.results_path+"/multinsc_{}epochs.pt".format(n_epochs)

		losses = []
		accuracies = []
		
		bat_per_epo = int(self.dataset.n_training_points / batch_size)
		n_steps = bat_per_epo * n_epochs
		
		for epoch in range(n_epochs):
			
			tmp_acc = []
			tmp_loss = []
			for i in range(bat_per_epo):
				
				# Select a minibatch
				X, _ , L = self.dataset.generate_mini_batches(batch_size)
		
				Xt = Variable(FloatTensor(X))
				Lt = Variable(LongTensor(L))
				optimizer.zero_grad()
				
				# Forward propagation: compute the output
				hypothesis_flat = self.mnsc(Xt)
				hypothesis = torch.reshape(hypothesis_flat, (batch_size, self.dataset.n_classes, self.dataset.n_outputs))

				# Computation of the cost J
				loss = loss_fnc(hypothesis, Lt) # <= compute the loss function
				
				# Backward propagation
				loss.backward() # <= compute the gradients
				
				# Update parameters (weights and biais)
				optimizer.step()
				
				# Print some performance to monitor the training
				glob_acc, _ = self.compute_accuracy(Lt, hypothesis)
				tmp_acc.append(glob_acc)
				tmp_loss.append(loss.item())   
			
			if epoch % 50 == 0:
				print("Epoch= {},\t loss = {:2.4f},\t accuracy = {}".format(epoch+1, tmp_loss[-1], tmp_acc[-1]))
				
			losses.append(np.mean(tmp_loss))
			accuracies.append(np.mean(tmp_acc))

		fig_loss = plt.figure()
		plt.plot(np.arange(n_epochs), losses, label="train")
		plt.tight_layout()
		plt.title("loss")
		fig_loss.savefig(self.results_path+"/loss_{}epochs.png".format(self.n_epochs))
		plt.close()

		fig_acc = plt.figure()
		plt.plot(np.arange(n_epochs), accuracies, label="train")
		plt.tight_layout()
		plt.title("accuracy")
		fig_acc.savefig(self.results_path+"/accuracy_{}epochs.png".format(self.n_epochs))
		plt.close()

		torch.save(self.mnsc, self.net_path)
		

	def load_trained_net(self, n_epochs):
		self.net_path = self.results_path+"/multinsc_{}epochs.pt".format(n_epochs)
		self.mnsc = torch.load(self.net_path)
		self.mnsc.eval()
		if cuda:
			self.mnsc.cuda()


	def generate_test_results(self):

		Xtest = Variable(FloatTensor(self.dataset.X_test_scaled))
		Ltest = Variable(LongTensor(self.dataset.L_test))
		test_preds_flat = self.mnsc(Xtest)
		test_preds = torch.reshape(test_preds_flat, (self.dataset.n_test_points, self.dataset.n_classes, self.dataset.n_outputs))

		#print("test labels = ", self.dataset.L_test, test_preds)
		global_test_accuracy, peroutput_test_accuracy = self.compute_accuracy(Ltest, test_preds)
		print("Global test accuracy: ", global_test_accuracy)
		print("Per-output test accuracy: ", peroutput_test_accuracy)

		os.makedirs(self.results_path, exist_ok=True)
		f = open(self.results_path+"/results.txt", "w")
		f.write("Global Test accuracy = ")
		f.write(str(global_test_accuracy))
		f.write("Per-output test accuracy = ")
		f.write(str(peroutput_test_accuracy))
		f.close()

