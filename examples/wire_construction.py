#!/usr/bin/env python

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os
import numpy as np
from ase.constraints import FixAtoms
from ase.calculators.espresso import Espresso
from supercell_builder import (Bulk, calculate_kpts, convert_miller_index,
                               cut_surface)
from qe_utils import read_qe_inp, read_qe_out
from nanoparticle_utils import (wire_construction, get_interact_len,
                                get_neighbor_atoms, create_interface_wire)

################################################################################
# BULK
################################################################################

bulk_type         = 'fcc'
input_bulk        = None
elements          = 'Rh'
lattice_constants = 3.863872
kpts_bulk         = (12, 12, 12)

bulk = Bulk(bulk_type         = bulk_type        ,
            input_bulk        = input_bulk       ,
            elements          = elements         ,
            lattice_constants = lattice_constants,
            kpts_bulk         = kpts_bulk        )

bulk = bulk.atoms

################################################################################
# WIRE
################################################################################

diameter = 8.

miller_indices   = [(1, 0, 0), (1, 1, 1)]
surface_energies = [       1.,        1.]

contact_index = (1, 1, 1)
adhesion_energy = 0.6

wire = wire_construction(bulk             = bulk            ,
                         diameter         = diameter        ,
                         miller_indices   = miller_indices  ,
                         surface_energies = surface_energies,
                         contact_index    = contact_index   ,
                         adhesion_energy  = adhesion_energy ,
                         vacuum           = 0.              )

del wire [[ a.index for a in wire if a.position[1] > 5.8 ]]

################################################################################
# CONTACT
################################################################################

input_data, pseudos, kpts, koffset = read_qe_inp('support/pw.inp')

support = read_qe_out('support/pw.out')

distance = 2.2

x_min = 6.
y_min = 18.
vacuum = 10.
stretch_max = 0.8

atoms = create_interface_wire(support     = support    ,
                              wire        = wire       , 
                              distance    = distance   ,
                              x_min       = x_min      ,
                              y_min       = y_min      ,
                              vacuum      = vacuum     ,
                              stretch_max = stretch_max)

atoms = cut_surface(atoms, surface_vectors = [[1., 0.], [2., 1.]])

indices = [ a.index for a in atoms if a.symbol == 'H' ]

constraints = atoms.constraints
constraints += [FixAtoms(indices = indices)]
atoms.set_constraint(constraints)

kpts = calculate_kpts(atoms      = atoms       ,
                      cell       = support.cell,
                      kpts       = kpts        ,
                      scale_kpts = 'xy'        )

################################################################################
# WRITE QUANTUM ESPRESSO INPUT
################################################################################

koffset = (0, 0, 1)

pseudos = {'Rh': 'Rh.UPF',
           'C' : 'C.UPF' ,
           'O' : 'O.UPF' ,
           'H' : 'H.UPF' ,
           'Al': 'Al.UPF'}

calc = Espresso(input_data       = input_data    ,
                pseudopotentials = pseudos       ,
                kpts             = kpts          ,
                koffset          = koffset       ,
                calculation      = 'relax'       ,
                restart_mode     = 'from_scratch',
                max_seconds      = 1700          )

calc.write_input(atoms)
os.rename('espresso.pwi', 'pw.inp')

################################################################################
# RUN QUANTUM ESPRESSO
################################################################################

run_qe = False

if run_qe is True:
    os.system("run pw -fs -n=2 -t=0.5")

################################################################################
# END
################################################################################
