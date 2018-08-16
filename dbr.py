import numpy as np
import pylab as pl
import timeit
import dbrprop


class DBRMirror:
    def __init__(self, data, incidentwavelength):
        if isinstance(data, str):
            data = np.loadtxt(data)
        self.n = data[:, 0]
        self.d = np.array(np.concatenate(([0], data[1:-1, 1], [0])))
        self.r = self.reflection_coeff(incidentwavelength)
        self.R = np.abs(self.r)**2
        self.GD = (self.r.imag * self.diff_omega(self.r.real, incidentwavelength) - self.r.real * self.diff_omega(self.r.imag, incidentwavelength))/self.R
        self.GDD = self.diff_omega(self.GD, incidentwavelength)

    def reflection_coeff(self, incidentwavelength):
        nm = (self.n[:-1] / self.n[1:])
        r = np.empty(incidentwavelength.size) + 0j
        i = 0
        for wavelength in incidentwavelength:
            phi = 2j * np.pi * self.n[:-1] * self.d[:-1] / wavelength
            transition_matrix = np.array([[0.5 * (1 + nm) * np.exp(-phi), 0.5 * (1 - nm) * np.exp(phi)],
                                       [0.5 * (1 - nm) * np.exp(-phi), 0.5 * (1 + nm) * np.exp(phi)]])

            general_transition_matrix = 1
            for k in range(self.n.size - 2, -1, -1):
                general_transition_matrix = np.dot(general_transition_matrix, transition_matrix[:, :, k])

            r[i] = - general_transition_matrix.item(2) / general_transition_matrix.item(3)
            i += 1
        return r

    def diff_omega(self, y, incidentwavelength):
        c = 29.9792458  # in nm/fs
        return (-incidentwavelength ** 2 / (2 * np.pi * c)) * np.gradient(y, incidentwavelength)

    # def diff_omega(self, y, incidentwavelength):
    #     res = np.zeros_like(y)
    #     res[1] = ((y[2] - y[1])/(incidentwavelength[2] - incidentwavelength[1]))
    #     res[-1] = ((y[-1] - y[-2])/(incidentwavelength[-1] - incidentwavelength[-2]))
    #     for n in range (2, y.__len__()-1):
    #         res[n] = (((y[n] - y[n-1]) / (incidentwavelength[n] - incidentwavelength[n-1]))
    #         + ((y[n+1] - y[n]) / (incidentwavelength[n+1] - incidentwavelength[n])))
    #     c = 29.9792458  # in nm/fs
    #     return - incidentwavelength ** 2 / (2 * np.pi * c) * res


start = timeit.default_timer()
wavelength = np.linspace(800, 1200, 1000)
NewMirror = DBRMirror("dbr.txt", wavelength)

stop = timeit.default_timer()
print(stop - start)

pl.figure()
pl.plot(wavelength, NewMirror.R)
pl.xlabel("$\lambda$")
pl.ylabel("R")
pl.xlim((950,1150))
pl.show(block=False)

pl.figure()
pl.plot(wavelength, NewMirror.GDD)
pl.xlabel("$\lambda$")
pl.ylabel("GDD [fs**2]")
pl.xlim((1000,1080))
pl.show()


