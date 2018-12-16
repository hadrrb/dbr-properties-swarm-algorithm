import numpy as np
import pylab as pl


class DBRMirror:
	def __init__(self, data, incidentwavelength):
		if isinstance(data, str):
			data = np.loadtxt(data)
		if data.ndim == 1:
			self.n = np.loadtxt("dbr.txt")[:, 0]
			self.d = data
		else:
			self.n = data[:, 0]
			self.d = np.array(np.concatenate(([0], data[1:-1, 1], [0])))
		self.r = self.reflection_coeff(incidentwavelength)
		self.R = np.abs(self.r)**2
		self.GD = (self.r.imag * self.diff_omega(self.r.real, incidentwavelength) - self.r.real * self.diff_omega(self.r.imag, incidentwavelength))/self.R
		self.GDD = self.diff_omega(self.GD, incidentwavelength)

	def reflection_coeff(self, incidentwavelength):
		nm = (self.n[:-1] / self.n[1:])
		general_transition_matrix = np.broadcast_to(np.eye(2, dtype= complex), (incidentwavelength.shape[0], 2, 2))

		for i in range(0, self.n.shape[0]-1):
			phi = 2j * np.pi * self.n[i+1] * self.d[i+1] / incidentwavelength
			transition_matrix = np.array([[0.5 * (1 + nm[i]) * np.exp(-phi), 0.5 * (1 - nm[i]) * np.exp(phi)],
										  [0.5 * (1 - nm[i]) * np.exp(-phi), 0.5 * (1 + nm[i]) * np.exp(phi)]])

			transition_matrix = np.swapaxes(transition_matrix, 2, 0)
			transition_matrix = np.swapaxes(transition_matrix, 2, 1)
			general_transition_matrix = np.array(list(map(lambda x, y: np.dot(x, y), general_transition_matrix, transition_matrix)))

		r = - general_transition_matrix[:,1,0] / general_transition_matrix[:,1,1]
		return r

	def diff_omega(self, y, incidentwavelength):
		c = 299.792458	# in nm/fs
		return (-incidentwavelength ** 2 / (2 * np.pi * c)) * np.gradient(y, incidentwavelength)



