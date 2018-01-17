from numpy import *
from pylab import *
import timeit


class DBRMirror:
    def __init__(self, ni, nbr): # ni - refractive indices as array, nbr - number of layers (must be even!)
        if nbr % 2 != 0: print("Number must be even! Program will give wrong results")
        self.nbr = nbr
        self.n = array(concatenate(([1], tile(ni, (self.nbr-2)//2), [ni[0]])))

    def reflection_coeff(self, incidentwavelength, braggwavelength):
        nm = (self.n[:-1] / self.n[1:])
        R = array([])
        for value in braggwavelength:
            d = array(concatenate(([0], 0.25 * value/self.n[1:-1], [0])))
            phi = 2j*pi*self.n[:-1]*d[:-1]/incidentwavelength
            transition_matrix = array([[0.5*(1+nm)*exp(-phi), 0.5*(1-nm)*exp(phi)],
                                  [0.5*(1-nm)*exp(-phi), 0.5*(1+nm)*exp(phi)]])

            general_transition_matrix = 1
            for k in range(self.nbr - 2,-1,-1):
                general_transition_matrix = dot(general_transition_matrix, transition_matrix[:, :, k])

            # print(general_transition_matrix)
            r = - general_transition_matrix.item(2) / general_transition_matrix.item(3)
            R = append(R, abs(r)**2)

            # t = linalg.det(general_transition_matrix) / general_transition_matrix.item(3)
            # T = n.item(N-1)/n.item(0) * abs(t)**2
            #
            # print(R)
            # print(T)
            # print(R+T)
        return R


start = timeit.default_timer()
wave_length = 0.98 * 10 ** (-6)
NewMirror = DBRMirror([3.0, 3.5], 42)
braggwavelength = array(linspace(0.68,1.28,100000)*10**(-6))
R = NewMirror.reflection_coeff(wave_length,braggwavelength)

stop = timeit.default_timer()
print(stop - start)

plot(braggwavelength, R)
xlabel("$\lambda$")
ylabel("R")
show()
