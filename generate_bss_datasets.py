from BSS_Architecture import *
from MyBikeSharing import *
import pickle

ARCH = "rombo"

if ARCH == "rombo":
	n_bikes = 100#50
	station_capacity = 25#12
	bss_arch = BSS_Architecture(n_bikes, station_capacity)

	departure_rates = (0.25, 0.2, 0.15)
	distances = (10,12)
	bss_arch.generate_rombo_arch(departure_rates, distances)
else: #triang
	n_bikes = 100#30
	station_capacity = 35#12
	bss_arch = BSS_Architecture(n_bikes, station_capacity)

	departure_rates = (0.25, 0.2, 0.15)
	distances = 10
	bss_arch.generate_triang_arch(departure_rates, distances)
	

nb_init_config = 2500
time_horizon = 10
nb_temp_points = 4*time_horizon#200

bss = MyBikeSharing(bss_arch.params_dict, plots_flag = False)

print("Generating initial configurations...")
bss.generate_set_initial_states(nb_init_config)


print("Generating SSA trajectories...")
n_trajs_per_config = 200
bss.SSA_simulation(n_trajs_per_config, time_horizon, nb_temp_points)
print("--Label SSA trajectories...")
bss.label_ssa_states()


ssa_filename = 'datasets/{}_ds_{}points_{}bikes_{}capacity_H={}_SSA.pickle'.format(ARCH, nb_init_config, n_bikes, station_capacity, time_horizon)
ssa_data_dict = {"x0": bss.init_configs, "labels": bss.SSA_labels}#"ssa_trajs": bss.SSA_trajs, 
with open(ssa_filename, 'wb') as handle:
	pickle.dump(ssa_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Generating MF trajectories...")
bss.MF_simulation(time_horizon, nb_temp_points)

print("--Label MF trajectories...")
bss.label_mf_states()

mf_filename = 'datasets/{}_ds_{}points_{}bikes_{}capacity_H={}_MF.pickle'.format(ARCH, nb_init_config, n_bikes, station_capacity, time_horizon)
mf_data_dict = {"x0": bss.init_configs, "labels": bss.MF_labels}#"mf_trajs": bss.MF_trajs,
with open(mf_filename, 'wb') as handle:
	pickle.dump(mf_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
