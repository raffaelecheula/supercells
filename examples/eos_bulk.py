#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os
import numpy as np
from ase.eos import EquationOfState
from qe_utils import read_eos_outputs

################################################################################
# EQUATION OF STATE
################################################################################

read_directories = True

plot_eos = False

npoints = 8
filename = 'pw.out'

if read_directories is True:

    os.chdir('eos')

    volumes, energies, cells = read_eos_outputs(npoints   = npoints ,
                                                filename  = filename,
                                                get_cells = True    )

    os.chdir('..')

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

cell_lengths = cells[0].lengths()/volumes[0]**(1./3.)*volume**(1./3.)

f = open('log.txt', 'w+')
f.write('cell length x = {0:14.7f} A\n'.format(cell_lengths[0]))
f.write('cell length y = {0:14.7f} A\n'.format(cell_lengths[1]))
f.write('cell length z = {0:14.7f} A\n'.format(cell_lengths[2]))
f.write('bulk energy   = {0:14.7f} eV\n'.format(energy))
f.write('bulk modulus  = {0:14.7f} eV/A^3\n'.format(bulkmodulus))
f.close()

if plot_eos is True:

    eos.plot(show = True)

################################################################################
# END
################################################################################
