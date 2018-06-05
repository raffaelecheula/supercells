#!/usr/bin/env python

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os
from ase import Atoms
from ase.calculators.espresso import Espresso
from supercell_builder import Bulk, Slab, Adsorbate, Vacancy
from supercell_utils import (read_qe_out, read_qe_inp, merge_supercells,
                             update_pseudos)

################################################################################
# MERGE SUPERCELLS
################################################################################

input_data, pseudos, kpts, koffset = read_qe_inp('one/pw.inp')
pseudos = update_pseudos(pseudos, 'two/pw.inp')

one = read_qe_out('one/pw.out')
two = read_qe_out('two/pw.out')

atoms = merge_supercells(one, two, vector = (0., 0.5), epsi = 5e-1)

################################################################################
# WRITE QUANTUM ESPRESSO INPUT
################################################################################

calc = Espresso(input_data       = input_data,
                pseudopotentials = pseudos,
                calculation      = 'relax',
                restart_mode     = 'from_scratch',
                max_seconds      = 1700,
                kpts             = kpts,
                koffset          = koffset)

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
