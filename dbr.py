from numpy import *
from pylab import *
import timeit


class DBRMirror:
    def __init__(self, data):
        if isinstance (data, str):
            data = loadtxt(data)
        self.n = data[:, 0]
        self.d = data[:, 1]


    def reflection_coeff(self, incidentwavelength):
        nm = (self.n[:-1] / self.n[1:])
        R = array([])
        for wavelength in incidentwavelength:
            phi = 2j*pi*self.n[:-1]*self.d[:-1]/wavelength
            transition_matrix = array([[0.5*(1+nm)*exp(-phi), 0.5*(1-nm)*exp(phi)],
                                  [0.5*(1-nm)*exp(-phi), 0.5*(1+nm)*exp(phi)]])

            general_transition_matrix = 1
            for k in range(self.n.size - 2, -1, -1):
                general_transition_matrix = dot(general_transition_matrix, transition_matrix[:, :, k])

            r = - general_transition_matrix.item(2) / general_transition_matrix.item(3)
            R = append(R, abs(r)**2)

        return R


start = timeit.default_timer()

data = np.array([
    [1.0,  0],
    [3.5,  74.0],
    [3.0,  92.5],
    [3.5,  74.0],
    [3.0,  92.5],
    [3.5,  74.0],
    [3.0,  92.5],
    [3.5, 0]
])

NewMirror = DBRMirror(data)

wavelength = array(linspace(600,2000,100000))

R = NewMirror.reflection_coeff(wavelength)

stop = timeit.default_timer()
print(stop - start)

plot(wavelength, R)
xlabel("$\lambda$")
ylabel("R")
show()
