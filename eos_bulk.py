#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import numpy as np
from ase.eos import EquationOfState
from qe_utils import read_eos_outputs

################################################################################
# EQUATION OF STATE
################################################################################

read_directories = True

plot_eos = True

npoints = 8
filename = 'pw.out'

if read_directories is True:

    volumes, energies = read_eos_outputs(npoints  = npoints ,
                                         filename = filename)

else:

    f = open('volumes_energies.txt', 'rU')
    
    energies = []
    volumes = []
    
    lines = f.readlines()
    
    for i in range(len(lines)-1):
        volumes.append(float(lines[i+1].split()[0]))
        energies.append(float(lines[i+1].split()[1]))
    f.close()

eos = EquationOfState(volumes, energies, eos = 'birchmurnaghan')
volume, energy, bulkmodulus = eos.fit()
latticeconstant = volume**(1./3.)

f = open('log.txt', 'w+')
f.write('lattice constant = {0:18.7f} A\n'.format(latticeconstant))
f.write('bulk energy      = {0:18.7f} eV\n'.format(energy))
f.write('bulk modulus     = {0:18.7f} eV/A^3\n'.format(bulkmodulus))
f.close()

if plot_eos is True:

    eos.plot(show = True)

################################################################################
# END
################################################################################
