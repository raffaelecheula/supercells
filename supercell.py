#!/usr/bin/python

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os
from addict import Dict
from ase import Atoms
from ase.calculators.espresso import Espresso
from supercell_builder import Bulk, Slab, Adsorbate, Vacancy

################################################################################
# BUILD SUPERCELL
################################################################################

CO = Atoms(('C', 'O'), ((0., 0., 0.), (0., 0., 1.2)))

bulk_type         = 'fcc'
input_bulk        = None
elements          = 'Rh'
lattice_constants = 3.863872
kpts_bulk         = (12, 12, 12)

input_slab        = None
miller_index      = '211'
surface_vectors   = 'automatic'
dimensions        = (2, 2)
layers            = 4
layers_fixed      = 3
symmetry          = 'asymmetric'
rotation_angle    = 'automatic'
scale_kpts        = 'xy'
adsorbates        = []
vacancies         = []
vacuum            = 10.
sort_atoms        = True

adsorbates.append(Adsorbate(CO, site = 'top', distance = 1.5))

bulk = Bulk(bulk_type         = bulk_type,
            input_bulk        = input_bulk,
            elements          = elements,
            lattice_constants = lattice_constants,
            kpts_bulk         = kpts_bulk)

slab = Slab(bulk              = bulk,
            input_slab        = input_slab,
            miller_index      = miller_index,
            surface_vectors   = surface_vectors,
            dimensions        = dimensions,
            layers            = layers,
            layers_fixed      = layers_fixed,
            symmetry          = symmetry,
            rotation_angle    = rotation_angle,
            scale_kpts        = scale_kpts,
            adsorbates        = adsorbates,
            vacancies         = vacancies,
            vacuum            = vacuum,
            sort_atoms        = sort_atoms)

atoms = slab.atoms
constraints = slab.constraints
kpts = slab.kpts
koffset = slab.koffset

################################################################################
# WRITE QUANTUM ESPRESSO INPUT
################################################################################

input_data = qe = Dict()

qe.outdir           = 'tmp'
qe.forc_conv_thr    = 1e-3
qe.etot_conv_thr    = 1e-4
qe.dipfield         = True

qe.ecutwfc          = 35.
qe.ecutrho          = 280.
qe.occupations      = 'smearing'
qe.smearing         = 'mv'
qe.degauss          = 0.001

qe.conv_thr         = 1e-6
qe.diagonalization  = 'david'
qe.diago_david_ndim = 2
qe.mixing_beta      = 0.2
qe.mixing_mode      = 'local-TF'
qe.electron_maxstep = 500

pseudos = {'Rh': 'Rh.UPF',
           'C' : 'C.UPF',
           'O' : 'O.UPF',
           'H' : 'H.UPF'}

calc = Espresso(input_data       = input_data,
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
