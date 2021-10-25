#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os
from ase.calculators.espresso import Espresso
from ase import Atoms
from ase.build import molecule
from ase.constraints import FixAtoms

################################################################################
# MOLECULE
################################################################################

atoms = Atoms('CO', positions = [[0., 0., 0.], [0., 0., 1.2]])

atoms.set_pbc(True)
atoms.center(vacuum = 10.)
atoms.set_constraint(FixAtoms(indices = [0]))

kpts  = (1, 1, 1)
koffset = (1, 1, 1)

################################################################################
# WRITE QUANTUM ESPRESSO INPUT
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
pseudos['H']  = 'Al.pbe-n-kjpaw_psl.1.0.0.UPF'

calc = Espresso(input_data       = pw_data,
                pseudopotentials = pseudos,
                kpts             = kpts,
                koffset          = koffset,
                calculation      = 'relax',
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
