from MyBikeSharing import *

n_stations = 5
n_bikes = 25
station_capacity = 10
dep = 0.1
dist = 10
departure_rates = dep*(np.ones((n_stations,n_stations))-np.eye(n_stations))
distances = dist*(np.ones((n_stations,n_stations))-np.eye(n_stations))

input_dict = {"n_stations": n_stations, "n_bikes": n_bikes, "station_capacity": station_capacity, 
				"departure_rates": departure_rates, "distances": distances}

nb_init_config = 10
n_trajs_per_config = 5
time_horizon = 40
nb_temp_points = 100

bss = MyBikeSharing(input_dict)

print("Generating initial configurations...")
init_states = bss.generate_set_initial_states(nb_init_config)

print("Generating MF trajectories...")
MF_trajs = bss.MF_simulation(time_horizon, nb_temp_points, init_states)

print("Generating SSA trajectories...")
SSA_trajs = bss.SSA_simulation(n_trajs_per_config, time_horizon, nb_temp_points, init_states)
