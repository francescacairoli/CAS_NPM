import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import copy

class MyBikeSharing():

    def __init__(self, params):
        self.n_stations = params["n_stations"]
        self.n_states = params["n_stations"]**2
        self.n_bikes = params["n_bikes"]
        self.station_capacity = params["station_capacity"]
        self.departure_rates = params["departure_rates"]
        self.distances = params["distances"]
        self.plots_flag = True
        self.station_idxs = np.eye(self.n_stations).flatten()
        self.type_reactions = 2
        self.n_transitions = self.type_reactions*self.n_states
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    def Ind1(self, z):
        if z > 0:
            return 1
        else:
            return 0

    def Ind2(self, z):
        if z < self.station_capacity:
            return 1
        else:
            return 0

    def Ind2_MF(self, z):
        if int(self.n_bikes*z) < self.station_capacity:
            return 1
        else:
            return 0
    def mean_field_ODE(self, t, x):

        X = np.reshape(x,(self.n_stations,self.n_stations))
        
        dx = np.zeros((self.n_stations,self.n_stations))
        for i in range(self.n_stations):
            for j in range(self.n_stations):
                if i != j:
                    dx[i,i] -= self.Ind1(X[i,i])*self.departure_rates[i,j]/self.n_bikes
                    dx[i,j] += (self.Ind1(X[i,i])*self.departure_rates[i,j]/self.n_bikes-X[i,j]*self.Ind2_MF(X[j,j])/self.distances[i,j])
                    dx[j,j] += X[i,j]*self.Ind2_MF(X[j,j])/self.distances[i,j]

        return dx.flatten()

    def generate_rnd_init_state(self):

        X0 = np.zeros((self.n_stations,self.n_stations))
        nb_available_bikes = self.n_bikes

        # sample nb of bikes in station
        for i in range(self.n_stations):
            X0[i,i] = np.random.randint(0, min(self.station_capacity,nb_available_bikes)+1)
            nb_available_bikes -= X0[i,i]

        already_visited_flag = np.zeros((self.n_stations,self.n_stations))

        # sample nb of bikes in transit
        while nb_available_bikes > 0:
            if nb_available_bikes == 1:
                xx = 1
            else:
                xx = np.random.randint(0, nb_available_bikes+1)
            tin_idx = np.random.randint(0,self.n_stations)
            tout_idx = np.random.randint(0,self.n_stations)
            if tin_idx != tout_idx and not already_visited_flag[tin_idx,tout_idx]:
                X0[tin_idx,tout_idx] = xx
                nb_available_bikes -= xx
                already_visited_flag[tin_idx,tout_idx] = 1

        return X0.flatten()

    def generate_set_initial_states(self, n_configs):

        return np.array([self.generate_rnd_init_state() for _ in range(n_configs)])

    def MF_simulation(self, final_time, n_time_points, init_configs):

        n_configs = len(init_configs)

        timestamp = np.linspace(0,final_time, n_time_points)
        MF_trajs = np.empty((n_configs,self.n_states, n_time_points))

        for j in range(n_configs):
            print('point = {}/{}'.format(j+1, n_configs))
            y0 = init_configs[j]/self.n_bikes
            sol = solve_ivp(self.mean_field_ODE, [0,final_time], y0, t_eval=timestamp, method='DOP853')
            MF_trajs[j] = sol.y
            
            if self.plots_flag:
                fig = plt.figure()
                c = 0
                for i in range(self.n_states):
                    if self.station_idxs[i]:
                        plt.plot(timestamp, MF_trajs[j,i], self.colors[c%len(self.colors)])
                        c += 1
                    #else:
                        #plt.plot(timestamp, MF_trajs[j,i],'--', self.colors[i%len(self.colors)])
                plt.title("Mean Field")
                plt.savefig("myplots/MF_trajs_{}.png".format(j))
                plt.close()

        return MF_trajs


    def evaluate_rates(self, state):

        state = np.reshape(state, (self.n_stations,self.n_stations))
        # R1: departure from station events
        R1 = np.zeros((self.n_stations,self.n_stations))
        R2 = np.zeros((self.n_stations,self.n_stations))
        for i in range(self.n_stations):
            for j in range(self.n_stations):
                if i != j:
                    R1[i,j] = self.departure_rates[i,j]*self.Ind1(state[i,i])
                    R2[i,j] = state[i,j]*self.Ind2(state[j,j])/self.distances[i,j]

        return np.concatenate((R1.flatten(), R2.flatten()))



    def get_update_vector(self, trans_index):

        update = np.zeros((self.n_stations, self.n_stations))
        if trans_index < self.n_states: # R1
            row = trans_index // self.n_stations
            col = trans_index % self.n_stations

            update[row,col] = 1
            update[row,row] = -1

        else:
            trans_index -= self.n_states
            row = trans_index // self.n_stations
            col = trans_index % self.n_stations

            update[row,col] = -1
            update[col,col] = +1

        return update.flatten()

    def SSA_simulation(self, n_trajs_per_config, final_time, n_time_points, init_configs):

        n_configs = len(init_configs)

        timestamp = np.linspace(0,final_time, n_time_points)
        SSA_trajs = np.empty((n_configs,n_trajs_per_config, self.n_states, n_time_points))

        for j in range(n_configs):
            print('point = {}/{}'.format(j+1, n_configs))

            for z in range(n_trajs_per_config):

                time = 0
                print_index = 1
                state = copy.deepcopy(init_configs[j])

                traj = np.zeros((n_time_points, self.n_states))
                traj[0, :] = state

                # main SSA loop
                
                while time < final_time:
                    # compute rates and total rate
                    rates = self.evaluate_rates(state)
                    # sanity check, to avoid negative numbers close to zero
                    rates[rates < 1e-14] = 0.0
                    total_rate = np.sum(rates)
                    # check if total rate is non zero.
                    if total_rate > 1e-14:
                        # if so, sample next time and next state and update state and time
                        trans_index = np.random.choice(self.n_transitions, p = rates / total_rate)
                        delta_time = np.random.exponential(1 / total_rate)
                        time += delta_time
                        update_vector = self.get_update_vector(trans_index)
                        state += update_vector
                    else:
                        # If not, stop simulation by skipping to final time
                        time = final_time
                    # store values in the output array
                    while print_index < n_time_points and timestamp[print_index] <= time:
                        traj[print_index, :] = state
                        print_index += 1          
                SSA_trajs[j,z] = traj.T

            if self.plots_flag:
                fig = plt.figure()
                for zz in range(n_trajs_per_config):
                    c = 0
                    for k in range(self.n_states):
                        if self.station_idxs[k]: # plot bikes in stations
                            plt.plot(timestamp, SSA_trajs[j,zz,k], self.colors[c%len(self.colors)])
                            c += 1
                        #else: # plot transitioning bikes
                            #plt.plot(timestamp, SSA_trajs[j,zz,k],'--', self.colors[k%len(self.colors)])
                plt.title("SSA")
                plt.savefig("myplots/SSA_trajs_{}.png".format(j))
                plt.close()            

        return SSA_trajs