from train_multi_nsc import *
from CP_MultiClassMultiOutput import *
from Dataset import *
import utility_functions as utils
import torch
from torch.autograd import Variable
import pickle
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--arch_name", type=str, default="triangular", help="Name of the architecture.")
parser.add_argument("--model_name", type=str, default="MF", help="Name of the model (first letters code).")
parser.add_argument("--time_horizon", type=int, default=10, help="Time horizon")
parser.add_argument("--do_refinement", type=bool, default=True, help="Flag: refine of the rejection rule.")
parser.add_argument("--nb_active_iteratiions", type=int, default=1, help="Number of active learning iterations.")
parser.add_argument("--nb_epochs", type=int, default=100, help="Number of epochs.")
parser.add_argument("--nb_epochs_active", type=int, default=400, help="Number of epochs in active learning.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.00001, help="Adam: learning rate")
parser.add_argument("--epsilon", type=float, default=0.05, help="CP significance level.")
parser.add_argument("--split_rate", type=float, default=50/65, help="adam: learning rate")
parser.add_argument("--pool_size_ref", type=int, default=25000, help="Size of the pool for the refinement step.")
parser.add_argument("--pool_size", type=int, default=50000, help="Size of the pool for one active learning step.")
parser.add_argument("--reinit_weights", type=bool, default=False, help="Flag: do reinitialize the weights in active learning steps.")

opt = parser.parse_args()
print(opt)

n_bikes = 100
capacity = 35

# Load the proper datasets: train, test, validation and calibration
trainset_fn = "datasets/{}_ds_1000points_{}bikes_{}capacity_H={}_{}.pickle".format(opt.arch_name, n_bikes, capacity, opt.time_horizon, opt.model_name)
testset_fn = "datasets/{}_ds_1000points_{}bikes_{}capacity_H={}_{}.pickle".format(opt.arch_name, n_bikes, capacity, opt.time_horizon, opt.model_name)
calibrset_fn = "datasets/{}_ds_2500points_{}bikes_{}capacity_H={}_{}.pickle".format(opt.arch_name, n_bikes, capacity, opt.time_horizon, opt.model_name)

dataset = Dataset(trainset_fn, testset_fn)
dataset.load_data()
'''
dataset.add_calibration_path(calibrset_fn)
dataset.load_calibration_data()

# Train and evalute the end-to-end approach (initial settings)
mnsc = Train_MultiNSC(opt.model_name, opt.arch_name, dataset)
start_time = time.time()
mnsc.train(opt.nb_epochs, opt.batch_size, opt.lr)
print("Multi NSC TRAINING TIME: ", time.time()-start_time)
print("----- Evaluate performances of the Multi NSC on the test set...")
mnsc.generate_test_results()

net_fnc = lambda inp: np.reshape(mnsc.mnsc(Variable(FloatTensor(inp))).cpu().detach().numpy(), (len(inp),dataset.n_classes,dataset.n_outputs))

# instance of the CP methods for classification (given the original calibration set)
cp = ICP_Classification(dataset=dataset, trained_model = net_fnc, mondrian_flag = False)
cp.set_calibration_scores()

# compute test validity and efficiency
print("----- Computing test CP validity...")
coverage = cp.compute_coverage(eps=opt.epsilon, inputs=dataset.X_test_scaled, outputs=dataset.L_test)
efficiency = cp.compute_efficiency()
print("Test empirical coverage: ", coverage, "Efficiency: ", efficiency)

print("----- Labeling correct/incorrect predictions...")
cal_errors = utils.label_correct_incorrect_pred(cp.cal_pred_lkh, dataset.L_cal) # shape (n_cal_points,n_outputs)
print("CALIBRATION ERRORS: ", cal_errors, np.sum(cal_errors[(cal_errors==1)]))
test_pred_lkh = net_fnc(dataset.X_test_scaled)
test_errors = utils.label_correct_incorrect_pred(test_pred_lkh, dataset.L_test) # shape (n_test_points,n_outputs)

print("----- Computing calibration confidence and credibility...")
cal_conf_cred = cp.compute_cross_confidence_credibility()

print("----- Training the query strategy on calibration data...")
kernel_type = 'rbf'

rej_fncs = []
for j in range(dataset.n_outputs):
	rej_fncs.append(utils.train_svc_query_strategy(kernel_type, cal_conf_cred[j], cal_errors[:,j]))


# apply rejection rule to the test set
start_time = time.time()
test_conf_cred = cp.compute_confidence_credibility(dataset.X_test_scaled)

test_pred_errors = []
for j in range(dataset.n_outputs):
	test_pred_errors.append(utils.apply_svc_query_strategy(rej_fncs[j], test_conf_cred[j]))
end_time = time.time()-start_time
print("Time to compute confid and cred over test set: ", end_time, "time per point: ", end_time/dataset.n_test_points)

rej_rates = []
for j in range(dataset.n_outputs):
	rej_rates.append(utils.compute_rejection_rate(test_pred_errors[j]))
print("----- Rejection rate = ", rej_rates)

nb_detections = []
nb_errors = []
detection_rates = []
for j in range(dataset.n_outputs):
	nb_detection_j, nb_errors_j, detection_rate_j = utils.compute_error_detection_rate(test_pred_errors[j], test_errors[:,j])
	nb_detections.append(nb_detection_j)
	nb_errors.append(nb_errors_j)
	detection_rates.append(detection_rate_j)
	print("----- Error detection rate for station {} = ".format(j), detection_rate_j, "({}/{})".format(nb_detection_j, nb_errors_j))
'''