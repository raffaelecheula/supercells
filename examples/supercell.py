#!/usr/bin/env python3

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import os
from ase import Atoms
from ase.build import molecule
from ase.calculators.espresso import Espresso
from supercells.builder import Bulk, Slab, Adsorbate, Vacancy

################################################################################
# BUILD SUPERCELL
################################################################################

"""
Set the parameters for the bulk phase.
"""
bulk_type         = 'fcc' # the symmetry of the bulk phase.
input_bulk        = None # a relaxed bulk structure file (not needed for fcc).
elements          = 'Rh' # the elements of the bulk phase.
lattice_constants = 3.8304732 # the lattice constant calculated by eos.py.
kpts_bulk         = (12, 12, 12) # the k points of the bulk.

"""
Set the parameters for the supercell.
"""
input_slab        = None # a relaxed slab structure file.
miller_index      = '111' # the miller index of the crystal facet: 'hkl'.
surface_vectors   = None # the vectors to cut the surface in x and y directions.
dimensions        = (1, 1) # repetitions of the slab in x and y directions.
layers            = 4 # repetitions of the slab in the z direction.
layers_fixed      = 3 # number of layers of the structure to not relax.
symmetry          = 'asymmetric' # the symmetry of the slab.
rotation_angle    = 'automatic' # rotation angle of the slab.
scale_kpts        = 'xy' # the vectors on which scale k points.
adsorbates        = [] # a vector containing the adsorbates.
vacancies         = [] # a vector containing the vacancies.
vacuum            = 10. # the vacuum to be added to the slab.
sort_atoms        = True # boolean to sort the atoms.

"""
Create Atoms objects representing the adsorbates.
"""
CO = Atoms(('C', 'O'), positions = [(0., 0., 0.), (0., 0., 1.2)])
H = Atoms(('H'), positions = [(0., 0., 0.)])

"""
Append the adsorbates to the adsorbates vector to be added to the slab.
"""
adsorbates.append(Adsorbate(CO, site = 'top', distance = 1.5))

"""
Create the Bulk object used to build the Slab.
"""
bulk = Bulk(bulk_type         = bulk_type        ,
            input_bulk        = input_bulk       ,
            elements          = elements         ,
            lattice_constants = lattice_constants,
            kpts_bulk         = kpts_bulk        )

"""
Create the Slab object.
"""
slab = Slab(bulk            = bulk           ,
            input_slab      = input_slab     ,
            miller_index    = miller_index   ,
            surface_vectors = surface_vectors,
            dimensions      = dimensions     ,
            layers          = layers         ,
            layers_fixed    = layers_fixed   ,
            symmetry        = symmetry       ,
            rotation_angle  = rotation_angle ,
            scale_kpts      = scale_kpts     ,
            adsorbates      = adsorbates     ,
            vacancies       = vacancies      ,
            vacuum          = vacuum         ,
            sort_atoms      = sort_atoms     )

"""
Extract the atoms, the k points and the offset of the k points.
"""
atoms = slab.atoms
kpts = slab.kpts
koffset = slab.koffset

################################################################################
# WRITE QUANTUM ESPRESSO INPUT
################################################################################

"""
Create the dictionary containing the parameters of the calulation.
"""
pw_data = {}

pw_data['outdir']           = 'tmp'
pw_data['forc_conv_thr']    = 1e-3
pw_data['etot_conv_thr']    = 1e-4
pw_data['dipfield']         = True

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

"""
Create the dictionary containing the pseudopotentials of the calulation.
"""
pseudos = {}

pseudos['Rh'] = 'Rh_ONCV_PBE-1.0.oncvpsp.UPF'
pseudos['C']  = 'C.pbe-n-kjpaw_psl.1.0.0.UPF'
pseudos['O']  = 'O.pbe-n-kjpaw_psl.0.1.UPF'
pseudos['H']  = 'H.pbe-rrkjus_psl.1.0.0.UPF'
pseudos['H']  = 'Al.pbe-n-kjpaw_psl.1.0.0.UPF'

"""
Create the Quantum Espresso calculator object.
"""
calc = Espresso(input_data       = pw_data,
                pseudopotentials = pseudos,
                kpts             = kpts,
                koffset          = koffset,
                calculation      = 'relax',
                restart_mode     = 'from_scratch',
                max_seconds      = 1700)

"""
Write the Quantum Espresso input file and rename it.
"""
calc.write_input(atoms)
os.rename('espresso.pwi', 'pw.inp')

################################################################################
# RUN QUANTUM ESPRESSO
################################################################################

"""
Run Quantum Espresso.
"""
run_qe = False

if run_qe is True:
    os.system("run pw -fs -n=2 -t=0.5")

################################################################################
# END
################################################################################
