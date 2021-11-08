#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os
import numpy as np
from ase.eos import EquationOfState
from ase.calculators.espresso import Espresso
from supercells.builder import Bulk
from shape.qe_utils import get_atom_list, assign_hubbard_U, create_eos_inputs

################################################################################
# BULK
################################################################################

bulk_type         = 'fcc'
input_bulk        = None
elements          = 'Rh'
lattice_constants = 3.83
kpts_bulk         = (12, 12, 12)

bulk = Bulk(bulk_type         = bulk_type        ,
            input_bulk        = input_bulk       ,
            elements          = elements         ,
            lattice_constants = lattice_constants,
            kpts_bulk         = kpts_bulk        )

atoms = bulk.atoms
kpts = bulk.kpts_bulk

koffset = (0, 0, 0)

################################################################################
# QUANTUM ESPRESSO VARIABLES
################################################################################

pw_data = {}

pw_data['outdir']           = 'tmp'
pw_data['forc_conv_thr']    = 1e-3
pw_data['etot_conv_thr']    = 1e-4

pw_data['ecutwfc']          = 35.
pw_data['ecutrho']          = 280.
pw_data['occupations']      = 'smearing'
pw_data['smearing']         = 'mv'
pw_data['degauss']          = 0.001

pw_data['conv_thr']         = 1e-6
pw_data['diagonalization']  = 'david'
pw_data['diago_david_ndim'] = 2
pw_data['mixing_beta']      = 0.2
pw_data['mixing_mode']      = 'local-TF'
pw_data['electron_maxstep'] = 500

pseudos = {}

pseudos['Rh'] = 'Rh_ONCV_PBE-1.0.oncvpsp.UPF'
pseudos['C']  = 'C.pbe-n-kjpaw_psl.1.0.0.UPF'
pseudos['O']  = 'O.pbe-n-kjpaw_psl.0.1.UPF'
pseudos['H']  = 'H.pbe-rrkjus_psl.1.0.0.UPF'
pseudos['Al'] = 'Al.pbe-n-kjpaw_psl.1.0.0.UPF'

calc = Espresso(input_data       = pw_data,
                pseudopotentials = pseudos,
                kpts             = kpts)

atoms.set_calculator(calc)

################################################################################
# WRITE ESPRESSO INPUT
################################################################################

write_esperesso_input = True

if write_esperesso_input is True:
    calc.write_input(atoms)
    os.rename('espresso.pwi', 'pw.inp')

################################################################################
# WRITE EOS INPUTS
################################################################################

write_eos_inputs = True

delta_x = 0.1
npoints = 8

run_cmd = None # "$BASH_SCRIPTS_DIR/run.sh pw -fs -n=2 -t=0.5"

if write_eos_inputs is True:
    create_eos_inputs(atoms   = atoms  ,
                      delta_x = delta_x,
                      npoints = npoints,
                      run_cmd = run_cmd)

################################################################################
# END
################################################################################
