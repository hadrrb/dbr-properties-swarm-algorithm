# <i>Bachelor Thesis:</i> Modelling of dispersive optical properties of DBRs using swarm algorithms

This repository contains the source code used to produce the results for the bachelor thesis (in Python3) in the main directory and the source code of the bachelor thesis itself (in LaTeX) in thesis folder. 

<u>Thesis text and code comments are in Polish.</u>

## Abstract

Swarm algorithms were used to control the dispersion in DBR mirrors in order to obtain ultrashort pulses. At the beginning, swarm algorithms were presented and tested if they really work. After that, the transition matrix method for determining the reflectivity and group delay dispersion for the given structure was shown, as well as the goal function. Then, 3 different approaches for representing the working variable of the algorithm for resolving the problem of finding DBR mirrors with possibly smallest value of group delay dispersion were presented, along with the corresponding results. The best results were observed in the first and easiest approach, that consisted in using layers' thickness, however it requires to start the calculations from a known, well optimised, structure. Observing the remaining results, it is clear that this method is working and the algorithm tends to generate better structures.

## Repository content description

The integral text of the thesis can be found in <a href="thesis/thesis.pdf">this pdf file</a> or compiled from <a href="thesis">source</a>. 

The project was originally run on a computer cluster at Lodz University of Technology. It consists of the following files:

- <a href="si">si</a>: definition of the job to be run on the cluster,
- <a href="dbr.py">dbr.py</a>: class defining a DBR mirror,
- <a href="si_mpi.py">si_mpi.py</a>: the main script containing an implementation of swarm algorithm,
- <a href="tools.py">tools.py</a>: function for creating a new directory,
- <a href="dbr.txt">dbr.txt</a>: the initial DBR structure,
- <a href="dbr_opt.txt">dbr_opt.txt</a>: the initial optimised DBR structure,
- <a href="args.txt">args.txt</a>: arguments to be passed to the main algorithm.