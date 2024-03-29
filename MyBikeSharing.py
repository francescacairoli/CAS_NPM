import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import copy

class MyBikeSharing():

    def __init__(self, params, plots_flag = False):
        self.n_stations = params["n_stations"]
        self.n_states = params["n_stations"]**2
        self.n_bikes = params["n_bikes"]
        self.station_capacity = params["station_capacity"]
        self.departure_rates = params["departure_rates"]
        self.distances = params["distances"]
        self.plots_flag = plots_flag
        self.station_idxs = np.eye(self.n_stations).flatten()
        self.type_reactions = 2
        self.n_transitions = self.type_reactions*self.n_states
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y']


    def label_mf_states(self):

        self.MF_labels = np.zeros((self.n_configs, self.n_stations, 2)) 
        # 2 labels: one-hot encoding [unsafe, safe]
        for i in range(self.n_configs):
            traj_i = np.reshape(np.round(self.n_bikes*self.MF_trajs[i]),(self.n_stations,self.n_stations,self.mf_n_time_points))
            for j in range(self.n_stations):
                Sj_traj = traj_i[j,j]
                if np.all((Sj_traj > 0)) and np.all((Sj_traj < self.station_capacity)):
                    self.MF_labels[i,j,1] = 1 # safe
                else:
                    self.MF_labels[i,j,0] = 1 # unsafe
            #print(i, self.MF_labels[i])

    def label_ssa_states(self):

        self.SSA_labels = np.zeros((self.n_configs, self.n_stations, 3)) 
        # 3 labels: one-hot encoding [unsafe, uncertain, safe]
        for i in range(self.n_configs):
            pool_trajs_i = self.SSA_trajs[i]
            lb_i = np.empty((self.n_states,self.ssa_n_time_points))
            ub_i = np.empty((self.n_states,self.ssa_n_time_points))
            for tt in range(self.ssa_n_time_points):
                for jj in range(self.n_states):
                    lb_i[jj,tt] = np.quantile(pool_trajs_i[:,jj,tt], q=0.025)
                    ub_i[jj,tt] = np.quantile(pool_trajs_i[:,jj,tt], q=0.925)

            LB_i = np.reshape(lb_i,(self.n_stations,self.n_stations,self.ssa_n_time_points))
            UB_i = np.reshape(ub_i,(self.n_stations,self.n_stations,self.ssa_n_time_points))

            for j in range(self.n_stations):
                Sj_lb_traj = LB_i[j,j]
                Sj_ub_traj = UB_i[j,j]
                if np.all((Sj_lb_traj > 0)) and np.all((Sj_ub_traj < self.station_capacity)):
                    self.SSA_labels[i,j,2] = 1 # safe
                elif np.any((Sj_ub_traj == 0)) and np.any((Sj_lb_traj == self.station_capacity)):
                    self.SSA_labels[i,j,0] = 1 # unsafe
                else:
                    self.SSA_labels[i,j,1] = 1 # uncertain/risky

            #print("LABELS for point {}:".format(i))
            #print(self.SSA_labels[i])

            if self.plots_flag:
                fig = plt.figure()
                plt.plot(self.ssa_timestamp, self.station_capacity*np.ones(self.ssa_n_time_points), '-.', c='k', label="capacity")
                c = 0
                for k in range(self.n_states):
                    if self.station_idxs[k]: # plot bikes in stations
                        #plt.plot(self.ssa_timestamp, lb_i[k], self.colors[c%len(self.colors)], label="S{}".format(c))
                        #plt.plot(self.ssa_timestamp, ub_i[k], self.colors[c%len(self.colors)])
                        plt.fill_between(self.ssa_timestamp, lb_i[k], ub_i[k], color=self.colors[c%len(self.colors)], alpha=0.2, label="S{}".format(c))
                        c += 1
                        #else: # plot transitioning bikes
                            #plt.plot(self.ssa_timestamp, SSA_trajs[j,zz,k],'--', self.colors[k%len(self.colors)])
                plt.title("stochastic")
                plt.legend(fontsize=14)
                plt.xlabel("time")
                plt.ylabel("nb. bikes")
                plt.tight_layout()
                plt.savefig("myplots/{}_SSA_bounds_{}stations_H={}.png".format(i, self.n_stations, self.ssa_final_time))
                plt.close() 


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

        already_visited_flag = np.eye(self.n_stations)

        # sample nb of bikes in transit
        while nb_available_bikes > 0:
            xx = np.random.randint(0, nb_available_bikes+1)
            tin_idx = np.random.randint(0,self.n_stations)
            tout_idx = np.random.randint(0,self.n_stations)
            if not already_visited_flag[tin_idx,tout_idx]:
                X0[tin_idx,tout_idx] = xx
                nb_available_bikes -= xx
                already_visited_flag[tin_idx,tout_idx] = 1
            if np.sum(already_visited_flag) == self.n_states:
                X0[tin_idx,tout_idx] = nb_available_bikes
                nb_available_bikes = 0
                
        return X0.flatten()

    def generate_set_initial_states(self, n_configs):

        self.n_configs = n_configs
        self.init_configs = np.empty((n_configs, self.n_states))
        for ii in range(n_configs):
            print("{}/{}".format(ii+1, n_configs))
            self.init_configs[ii] = self.generate_rnd_init_state()

    def MF_simulation(self, final_time, n_time_points):

        self.mf_final_time = final_time
        self.mf_n_time_points = n_time_points

        self.mf_timestamp = np.linspace(0,final_time, n_time_points)
        MF_trajs = np.empty((self.n_configs,self.n_states, n_time_points))

        for j in range(self.n_configs):
            print('point = {}/{}'.format(j+1, self.n_configs))
            y0 = self.init_configs[j]/self.n_bikes
            sol = solve_ivp(self.mean_field_ODE, [0,final_time], y0, t_eval=self.mf_timestamp, method='DOP853')
            MF_trajs[j] = sol.y
            
            if self.plots_flag:
                fig = plt.figure()
                plt.plot(self.mf_timestamp, self.station_capacity*np.ones(self.mf_n_time_points), '-.', c='k',label="capacity")
                c = 0
                for i in range(self.n_states):
                    if self.station_idxs[i]:
                        plt.plot(self.mf_timestamp, np.round(self.n_bikes*MF_trajs[j,i]), self.colors[c%len(self.colors)], label="S{}".format(c))
                        c += 1
                    #else:
                        #plt.plot(self.mf_timestamp, MF_trajs[j,i],'--', self.colors[i%len(self.colors)])
                plt.title("deterministic")
                plt.legend(fontsize=14)
                plt.xlabel("time")
                plt.ylabel("nb. bikes")
                plt.tight_layout()
                plt.savefig("myplots/{}_MF_trajs_{}stations_H={}.png".format(j, self.n_stations, final_time))
                plt.close()

        self.MF_trajs = MF_trajs


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

    def SSA_simulation(self, n_trajs_per_config, final_time, n_time_points):

        self.ssa_final_time = final_time
        self.ssa_n_time_points = n_time_points
        
        self.n_trajs_per_config = n_trajs_per_config

        self.ssa_timestamp = np.linspace(0,final_time, n_time_points)
        SSA_trajs = np.empty((self.n_configs,n_trajs_per_config, self.n_states, n_time_points))

        for j in range(self.n_configs):
            print('point = {}/{}'.format(j+1, self.n_configs))

            for z in range(n_trajs_per_config):

                time = 0
                print_index = 1
                state = copy.deepcopy(self.init_configs[j])

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
                    while print_index < n_time_points and self.ssa_timestamp[print_index] <= time:
                        traj[print_index, :] = state
                        print_index += 1          
                SSA_trajs[j,z] = traj.T

            if self.plots_flag:
                fig = plt.figure()
                plt.plot(self.ssa_timestamp, self.station_capacity*np.ones(self.ssa_n_time_points), '-.', c='k', label="capacity")
                for zz in range(n_trajs_per_config):
                    c = 0
                    for k in range(self.n_states):
                        if self.station_idxs[k]: # plot bikes in stations
                            if zz == 0:
                                plt.plot(self.ssa_timestamp, SSA_trajs[j,zz,k], self.colors[c%len(self.colors)], label="S{}".format(c))
                            else:
                                plt.plot(self.ssa_timestamp, SSA_trajs[j,zz,k], self.colors[c%len(self.colors)])
                            c += 1
                        #else: # plot transitioning bikes
                            #plt.plot(self.ssa_timestamp, SSA_trajs[j,zz,k],'--', self.colors[k%len(self.colors)])
                plt.title("stochastic")
                plt.legend(fontsize=14)
                plt.xlabel("time")
                plt.ylabel("nb. bikes")
                plt.tight_layout()
                plt.savefig("myplots/{}_SSA_trajs_{}stations_H={}.png".format(j, self.n_stations, final_time))
                plt.close()            

        self.SSA_trajs = SSA_trajs