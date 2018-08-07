from numpy import *
from pylab import *
import timeit
import dbrprop

class DBRMirror:
    def __init__(self, data):
        if isinstance(data, str):
            data = loadtxt(data)
        self.n = data[:, 0]
        self.d = array(concatenate(([0], data[1:-1, 1], [0])))

    def reflection_coeff(self, incidentwavelength):
        nm = (self.n[:-1] / self.n[1:])
        R = array([])
        for wavelength in incidentwavelength:
            phi = 2j * pi * self.n[:-1] * self.d[:-1] / wavelength
            transition_matrix = array([[0.5 * (1 + nm) * exp(-phi), 0.5 * (1 - nm) * exp(phi)],
                                       [0.5 * (1 - nm) * exp(-phi), 0.5 * (1 + nm) * exp(phi)]])

            general_transition_matrix = 1
            for k in range(self.n.size - 2, -1, -1):
                general_transition_matrix = dot(general_transition_matrix, transition_matrix[:, :, k])

            r = - general_transition_matrix.item(2) / general_transition_matrix.item(3)
            # R = append(R, abs(r)**2)
            R = append(R, r)

        phi = arctan(imag(R) / real(R))
        c = 29.9792458  # in nm/fs
        GDD = incidentwavelength ** 2 / (2 * pi * c) * gradient(incidentwavelength ** 2 / (2 * pi * c) * gradient(phi))
        return GDD


start = timeit.default_timer()

# data = array([
#     [1.0, inf],
#     [3.5, 74.0],
#     [3.0, 92.5],
#     [3.5, 74.0],
#     [3.0, 92.5],
#     [3.5, 74.0],
#     [3.0, 92.5],
#     [3.5, inf]
# ])

New = dbrprop.DBRprop()
NewMirror = DBRMirror(New.data)

wavelength = array(linspace(400, 4000, 1000))

gdd = NewMirror.reflection_coeff(wavelength)

stop = timeit.default_timer()
print(stop - start)

plot(wavelength, gdd)
xlabel("$\lambda$")
ylabel("GDD [fs**2]")
show()
