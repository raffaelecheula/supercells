#!/usr/bin/env python

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os
from ase.calculators.espresso import Espresso
from supercell_builder import calculate_kpts
from supercell_utils import (read_qe_inp, read_qe_out, create_interface_slab,
                             update_pseudos)

################################################################################
# CREATE INTERFACE SLAB
################################################################################

input_data, pseudos, kpts, koffset = read_qe_inp('support/pw.inp')
pseudos = update_pseudos(pseudos, 'surface/pw.inp')

support = read_qe_out('support/pw.out')
surface = read_qe_out('surface/pw.out')

distance = 1.5

atoms = create_interface_slab(slab_a           = support,
                              slab_b           = surface, 
                              distance         = distance,
                              symmetry         = 'inversion',
                              adapt_dimensions = True,
                              area_max         = 40.,
                              nmax             = 7,
                              stretch_max      = 0.15,
                              toll_rotation    = 5e-2)

kpts = calculate_kpts(atoms      = atoms,
                      cell       = support.cell,
                      kpts       = kpts,
                      scale_kpts = 'xy')

################################################################################
# WRITE QUANTUM ESPRESSO INPUT
################################################################################

calc = Espresso(input_data       = input_data,
                pseudopotentials = pseudos,
                kpts             = kpts,
                koffset          = koffset,
                calculation      = 'vc-relax',
                restart_mode     = 'from_scratch',
                max_seconds      = 1700)

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
