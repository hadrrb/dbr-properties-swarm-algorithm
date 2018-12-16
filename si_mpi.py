#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# import pylab as pl
import matplotlib.pyplot as pl
from pylab import *
from mpl_toolkits import mplot3d
import dbr
import timeit
from tools import mkdir
import sys
from mpi4py.MPI import COMM_WORLD as mpi
from matplotlib.backends.backend_pdf import PdfPages

size = mpi.Get_size()
rank = mpi.Get_rank()

args = np.loadtxt("args.txt")
job = int(sys.argv[1])
path = str(sys.argv[2]) # folder do którego zostaną zapisane wyniki

m =	0		# liczba miejsc wybranych spośród n odwiedzonych miejsc i gorszych od e
nep = 0		# liczba pszczół przydzielonych do najlepszych e miejsc
e =	0		# liczba najlepszych miejsc spośród wyselekcjonowanych n miejsc
nsp = 0		# liczba pszczół zrekrutowanych do m wybranych miejsc
n = int(args[job][0]) # liczba zwiadowców
Neighb = args[job][1]  # odległość przeszukiwania od najlepszego aktualnego rozwiązania
ngh = args[job][2] # odległość przeszukiwania od rozwiązania bazowego przy przeszukiwaniu sąsiedztwa
qual = args[job][3] # wskaźnik jakości
best = args[job][4]  # najlepsze z najlepszych
praw = args[job][5] # prawdopodobieństwo znalezienia lepszego rozwiązania w sąsiedztwie rozw elitarnych
w = np.zeros(4)
w[0]= args[job][6]	# współczynnik określający wpływ R na rozwiązanie
w[1] = args[job][7] # współczynnik określający wpływ średniej wartośći GDD na rozwiązanie
w[2] = args[job][8] # współczynnik określający wpływ odchylenia GDD od wartości średniej
w[3] = args[job][9] # współczynnik określający wpływ rozpiętości zakresu wartości GDD
s = args[job][10] # parametr określający zakres badanych długości fali
eps = 0.001

time_limit = 10 * 3600
wavelength = np.linspace(1020, 1060, 1000)
n_dbr = np.loadtxt("dbr.txt")[:, 0]
l = 1040 # badana długość fali

def opticalpath(d):
	op = np.zeros(n_dbr.size-2)
	nd = n_dbr*d
	op[0:-1:2] = 2*(nd[1:-2:2] + nd[2:-1:2])/(l*s) - 1/s + 0.5
	op[1::2] = nd[1:-2:2]/(nd[1:-2:2] + nd[2:-1:2])
	return op
	
def dlen(op):
	d = np.zeros(n_dbr.shape);
	c = op[0:-1:2]
	h = op[1::2]
	d[1:-2:2] = (l*h*(1+s*(c-0.5))/n_dbr[1]*0.5).clip(min=5)
	d[2:-1:2] = (l*(1- h)*(1+s*(c-0.5))/(n_dbr[2]*2)).clip(min=5)
	return d
	
def goal(x, w):
	if x.ndim > 1:
		res = []
		for i in range (0, x.shape[0]):
			res = np.append(res, goal(x[i], w))
		return res
	else:
		x = dlen(x)
		new = dbr.DBRMirror(x, wavelength)
		par1 = -(new.R < 0.98).sum()
		par2 = -np.mean(new.GDD)
		par3 = -np.std(new.GDD)
		par4 = -np.ptp(new.GDD)  # difference between max and min
		return w[0] * par1 + w[1] * par2 + w[2] * par3 + w[3] * par4 
			
def gen(a, b, n):
	a[0:-2:2] = (a[0:-2:2]).clip(min=0, max=1)
	a[1:-1:2] = (a[1:-1:2]).clip(min=eps, max=1 - eps)
	b[0:-2:2] = (b[0:-2:2]).clip(min=0, max=1)
	b[1:-1:2] = (b[1:-1:2]).clip(min=eps, max=1 - eps)
	return (np.random.rand(n, a.size)*(b-a) + a)

def neighb(x, n, ng):
	return np.vstack((x, gen(x - ng, x + ng, n-1)))


if rank == 0:
	solO = opticalpath(np.loadtxt("dbr.txt")[:, 1])
	sol = neighb(solO, n, Neighb)
	wyn = solO
	cel_wyn = goal(wyn, w)
	mkdir(path)
else:
	sol = np.array([])
	
sol = mpi.bcast(sol, root = 0)
j = 0	
start = timeit.default_timer()
stop = start
end = 500
mpi.Barrier()	

while (j <= end) & ((stop - start) < time_limit):
	best_sol = np.array([])
	good_sol = np.array([])
	m = 0
	e = 0
	cel = np.array([])
	
	period = n // size
	full_range = np.linspace(0, size, size+1) * period
	if full_range[-1] < n:
		full_range[-1] = n
		
	tempsol = sol[int(full_range[rank]):int(full_range[rank+1])]
	cel = goal(tempsol, w)	
	mpi.Barrier()
	
	cel = mpi.gather(cel)
	if rank == 0:
		cel = concatenate(cel)
		sort = np.argsort(cel)
		cel = cel[sort[::-1]]
		sol = sol[sort[::-1]]

	mpi.Barrier()

	sol = mpi.bcast(sol, root = 0)
	cel = mpi.bcast(cel, root = 0)
	e = int(n*best)
	m = int(n*qual)

	best_sol = sol[0:e-1]
	good_sol = sol[e:m-1]
	left = (n - m - e)			
	nep = left * praw + e
	nsp = left*(1-praw) + m
	
	period = e // size
	
	full_range = np.linspace(0, size, size+1) * period
	if full_range[-1] < e:
		full_range[-1] = e

	sol = np.array([])
	cel = np.array([])

	best_sol = best_sol[int(full_range[rank]):int(full_range[rank+1])]
	for i in best_sol:
		if best_sol.ndim == 1:
			res = neighb(best_sol, np.int(nep/e),ngh)
		else:
			res = neighb(i, np.int(nep/e),ngh)
		if not sol.any():
			celres = goal(res, w)
			cmax = np.argmax(celres)
			cel = celres[cmax]
			sol = res[cmax]
		else:
			celres = goal(res, w)
			cmax = np.argmax(celres)
			cel = np.append(cel, celres[cmax])
			sol = np.vstack((sol, res[cmax]))
		if best_sol.ndim == 1:
			break
			
	period = (m-e) // size
			
	full_range = np.linspace(0, size, size+1) * period +e
	if full_range[-1] < m:
		full_range [-1] = m

	good_sol = good_sol[int(full_range[rank]):int(full_range[rank+1])]
	for i in good_sol:
		if good_sol.ndim == 1:
			res = neighb(good_sol, np.int(nsp/m),ngh)
		else:
			res = neighb(i, np.int(nsp/m),ngh)	  
		if not sol.any():
			celres = goal(res, w)
			cmax = np.argmax(celres)
			cel = celres[cmax]
			sol = res[cmax]
		else:
			celres = goal(res, w)
			cmax = np.argmax(celres)
			cel = np.append(cel, celres[cmax])
			sol = np.vstack((sol, res[cmax]))
		if good_sol.ndim == 1:
			break
	
	mpi.Barrier()
	stop = timeit.default_timer()
	
	cel = mpi.gather(cel)
	sol = mpi.gather(sol)
	
	if rank == 0:
		cel = concatenate(cel)
		sol = np.vstack(sol)
		sort = np.argsort(cel)
		cel = cel[sort[::-1]]
		sol = sol[sort[::-1]]

		wyn = sol[0]
		cel_wyn = cel[0]
		if (j < end) & (sol.size < n) & ((stop - start) < time_limit):
			if m+e > n*0.3:
				best = qual * best
				qual = qual * qual
				ngh = ngh/2
			sol = np.vstack((sol, neighb(sol0, n-sol.size, Neighb)))

		wyn = dlen(wyn)
		np.savetxt(path + "/result" + str(j) +".txt", wyn) 

		result = dbr.DBRMirror(wyn, wavelength)
		med = dbr.DBRMirror(dlen(sol[sol.shape[0]//2]), wavelength)
		par1 = (result.R > 0.98).sum()
		par2 = -np.mean(result.GDD)
		par3 = -np.std(result.GDD)
		par4 = -np.ptp(result.GDD)
		
		print("\nIteracja = " + str(j))
		print("Cel = " + str(cel[0]))
		print("par1 = " + str(par1))
		print("par2 = " + str(par2))
		print("par3 = " + str(par3))
		print("par4 = " + str(par4))
		
		if(j%10==0):
			plik = path + "/result" + str(j) +'.pdf'
			with PdfPages(plik) as pdf:
				f = pl.figure()
				pl.clf()
				pl.axis('off')
				np.set_printoptions(suppress=True)
				txt = r"$\bf{Argumenty:}$" +'\nn = ' + str(n) + '\nNeighb = ' + str(Neighb)+ '\nngh = ' + str(ngh) + '\nqual = ' + str(qual) +'\nbest = ' + str(best) + '\npraw = ' + str(praw) +'\nw[0] = ' + str(w[0]) + '\nw[1] = ' + str(w[1]) + '\nw[2] = ' + str(w[2]) + '\nw[3] = ' + str(w[3]) + '\ns = ' + str(s) +'\n\n' + r"$\bf{Iteracja}$ " + r"$\bf{" + str(j) + "}$"
				pl.text(0.5,0.5, txt, transform=f.transFigure, size=10, ha="center")
				pdf.savefig()
				pl.close()
				
				f = pl.figure()
				pl.plot(wavelength, result.R, label='wynik')
				pl.plot(wavelength, med.R, label='mediana')
				pl.xlabel("$\lambda$")
				pl.ylabel("R")
				pl.legend(loc='upper right')
				pl.title("R najlepszego rozwiązania")
				pdf.savefig()
				pl.close()

				f = pl.figure()
				pl.plot(wavelength, result.GDD, label='wynik')
				pl.plot(wavelength, med.GDD, label='mediana')
				pl.xlabel("$\lambda$")
				pl.ylabel("GDD [fs$^2$]")
				pl.legend(loc='upper right')
				pl.title("GDD najlepszego rozwiązania")
				pdf.savefig()
				pl.close()
				
				f = pl.figure()
				pl.bar(np.linspace(0, result.d.size - 1, result.d.size), result.d, color = ['blue','black'])
				pl.title("Grubości warstw dla najlepszego rozwiązania")
				pdf.savefig()
				pl.close()
				
				f = pl.figure()
				pl.bar(np.linspace(0, med.d.size - 1, med.d.size), med.d, color = ['blue','black'])
				pl.title("Grubości warstw dla średniego rozwiązania")
				pdf.savefig()
				pl.close()
				
		if(j==0):
			pdf_wyn = PdfPages("wyn" + str(job)+ '.pdf') # do stworzenia pdf z wynikami z pierwszej i ostatniej iteracji
			
			f = pl.figure()
			pl.clf()
			pl.axis('off')
			np.set_printoptions(suppress=True)
			txt = r"$\bf{Argumenty:}$" +'\nn = ' + str(n) + '\nNeighb = ' + str(Neighb)+ '\nngh = ' + str(ngh) + '\nqual = ' + str(qual) +'\nbest = ' + str(best) + '\npraw = ' + str(praw) +'\nw[0] = ' + str(w[0]) + '\nw[1] = ' + str(w[1]) + '\nw[2] = ' + str(w[2]) + '\nw[3] = ' + str(w[3]) + '\ns = ' + str(s)
			pl.text(0.5,0.5, txt, transform=f.transFigure, size=10, ha="center")
			pdf_wyn.savefig()
			pl.close()
			
			f = pl.figure()
			pl.plot(wavelength, result.R, label='wynik')
			pl.plot(wavelength, med.R, label='mediana')
			pl.xlabel("$\lambda$")
			pl.ylabel("R")
			pl.legend(loc='upper right')
			pl.title(r"$\bf{Iteracja}$ " + r"$\bf{" + str(j) + "}$" + "\nR najlepszego rozwiązania")
			pdf_wyn.savefig()
			pl.close()

			f = pl.figure()
			pl.plot(wavelength, result.GDD, label='wynik')
			pl.plot(wavelength, med.GDD, label='mediana')
			pl.xlabel("$\lambda$")
			pl.ylabel("GDD [fs$^2$]")
			pl.legend(loc='upper right')
			pl.title("GDD najlepszego rozwiązania")
			pdf_wyn.savefig()
			pl.close()
			
			f = pl.figure()
			pl.bar(np.linspace(0, result.d.size - 1, result.d.size), result.d, color = ['blue','black'])
			pl.title("Grubości warstw dla najlepszego rozwiązania")
			pdf_wyn.savefig()
			pl.close()
			
			f = pl.figure()
			pl.bar(np.linspace(0, med.d.size - 1, med.d.size), med.d, color = ['blue','black'])
			pl.title("Grubości warstw dla średniego rozwiązania")
			pdf_wyn.savefig()
			pl.close()
			
		if(j==end):
			f = pl.figure()
			pl.plot(wavelength, result.R, label='wynik')
			pl.plot(wavelength, med.R, label='mediana')
			pl.xlabel("$\lambda$")
			pl.ylabel("R")
			pl.legend(loc='upper right')
			pl.title(r"$\bf{Iteracja}$ " + r"$\bf{" + str(j) + "}$" + "\nR najlepszego rozwiązania")
			pdf_wyn.savefig()
			pl.close()

			f = pl.figure()
			pl.plot(wavelength, result.GDD, label='wynik')
			pl.plot(wavelength, med.GDD, label='mediana')
			pl.xlabel("$\lambda$")
			pl.ylabel("GDD [fs$^2$]")
			pl.legend(loc='upper right')
			pl.title("GDD najlepszego rozwiązania")
			pdf_wyn.savefig()
			pl.close()
			
			f = pl.figure()
			pl.bar(np.linspace(0, result.d.size - 1, result.d.size), result.d, color = ['blue','black'])
			pl.title("Grubości warstw dla najlepszego rozwiązania")
			pdf_wyn.savefig()
			pl.close()
			
			f = pl.figure()
			pl.bar(np.linspace(0, med.d.size - 1, med.d.size), med.d, color = ['blue','black'])
			pl.title("Grubości warstw dla średniego rozwiązania")
			pdf_wyn.savefig()
			pl.close()
			pdf_wyn.close()
		
	j = j+1
	sol = mpi.bcast(sol, root = 0)
	mpi.Barrier()	
	