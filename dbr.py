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
        r = np.empty(incidentwavelength. size) + 0j
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
        c = 299.792458  # in nm/fs
        return (-incidentwavelength ** 2 / (2 * np.pi * c)) * np.gradient(y, incidentwavelength)


start = timeit.default_timer()
wavelength = np.linspace(800, 1200, 1000)
NewMirror = DBRMirror("dbr.txt", wavelength)

stop = timeit.default_timer()
print(stop - start)
sR = np.loadtxt("s245-R.txt", unpack=True)
sGD = np.loadtxt("s245-GD.txt", unpack=True)
sGDD = np.loadtxt("s245-GDD.txt", unpack=True)


f = pl.figure()
pl.plot(wavelength, NewMirror.R)
pl.plot(*sR)
pl.xlabel("$\lambda$")
pl.ylabel("R")
pl.xlim(950, 1150)
pl.show(block=False)
f.canvas.set_window_title("R")

f = pl.figure()
pl.plot(wavelength, NewMirror.GD)
pl.plot(*sGD)
pl.xlabel("$\lambda$")
pl.ylabel("GD [fs]")
pl.xlim(950, 1150)
f.canvas.set_window_title("GD")

f = pl.figure()
pl.plot(wavelength, NewMirror.GDD)
pl.plot(*sGDD)
pl.xlabel("$\lambda$")
pl.ylabel("GDD [fs$^2$]")
pl.xlim(1000, 1080)
pl.ylim(-1000, 0)
f.canvas.set_window_title("GDD")

pl.show()


