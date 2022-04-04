import numpy as np

class BSS_Architecture():

	def __init__(self, n_bikes, station_capacity):

		self.n_bikes = n_bikes
		self.station_capacity = station_capacity

	def generate_triang_arch(self, departure_rates, dist):

		'''
		3 equidistant stations (dist)
		'''

		self.n_stations = 3

		dep0, dep1, dep2 = departure_rates
		
		u = np.ones(self.n_stations)
		dep_matrix = np.vstack((dep0*u,dep1*u,dep2*u))-np.diag([dep0,dep1,dep2])
		self.departure_rates = dep_matrix
		print("departures_matrix = \n", dep_matrix)

		self.distances = dist*(np.ones((self.n_stations,self.n_stations))-np.eye(self.n_stations))
		print("distances_matrix = \n", self.distances)

		self.params_dict = {"n_stations": self.n_stations, "n_bikes": self.n_bikes, "station_capacity": self.station_capacity, 
				"departure_rates": self.departure_rates, "distances": self.distances}



	def generate_rombo_arch(self, departure_rates, distances):

		'''
		TODO: check that:
			dep_center > dep_near > dep_farv
			d_near     <   d_far  < d_diag

		# S0 = center
		# S1,S2 = near
		# S3,S4 = far
		'''

		self.n_stations = 5

		dep_center, dep_near, dep_far = departure_rates
		d_near, d_far = distances

		d_diag = np.sqrt(d_near**2+d_far**2)

		u = np.ones(self.n_stations)
		dep_matrix = np.vstack((dep_center*u,dep_near*u,dep_near*u,dep_far*u,dep_far*u))-np.diag([dep_center,dep_near,dep_near,dep_far,dep_far])
		print("departures_matrix = \n", dep_matrix)

		dist_matrix = np.zeros((self.n_stations,self.n_stations))
		dist_matrix[0] = [0,d_near,d_near,d_far,d_far]
		dist_matrix[1] = [d_near, 0, 2*d_near, d_diag, d_diag]
		dist_matrix[2] = [d_near, 2*d_near, 0, d_diag, d_diag]
		dist_matrix[3] = [d_far, d_diag, d_diag, 0, 2*d_far]
		dist_matrix[4] = [d_far, d_diag, d_diag, 2*d_far, 0]
		print("distances_matrix = \n", dist_matrix)

		self.departure_rates = dep_matrix
		self.distances = dist_matrix

		self.params_dict = {"n_stations": self.n_stations, "n_bikes": self.n_bikes, "station_capacity": self.station_capacity, 
				"departure_rates": self.departure_rates, "distances": self.distances}


