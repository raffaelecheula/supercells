#!/usr/bin/python

################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import division, print_function
from write_pwscf import PWscf
from supercell_builder import *

################################################################################
#                                SUPERCELL
################################################################################

CO = [['C', 0., 0., 0.], ['O', 0., 0., 1.]]

sc = Supercell()

sc.bulk_type         = 'fcc'
sc.elements          = 'Rh'
sc.lattice_constants = 3.863872

sc.input_slab        = None
sc.miller_index      = '100'
sc.surface_vectors   = 'automatic'
sc.dimensions        = (2, 2)

sc.layers            = 5
sc.layers_fixed      = 4
sc.symmetry          = None
sc.cut_top           = None
sc.cut_bottom        = None
sc.vacuum            = 10.

sc.adsorbates        = (CO, 'hollow', 1, 1.5)
sc.vacancies         = None
sc.units             = 'final_cell'
sc.break_sym         = False
sc.rotation_angle    = 'automatic'

sc.k_points          = [12, 12, 1]
sc.scale_kp          = 'xy'
sc.epsi              = 1e-4

system   = sc.create_system()
height   = sc.height
k_points = sc.k_points

################################################################################
#                         QUANTUM ESPRESSO VARIABLES
################################################################################

qe = PWscf()

# CONTROL
qe.control.calculation           = 'relax'
qe.control.restart_mode          = 'from_scratch'
qe.control.forc_conv_thr         = 1.0e-3
qe.control.etot_conv_thr         = 1.0e-4
qe.control.max_seconds           = 1700
qe.control.prefix                = 'calc'
qe.control.outdir                = './tmp/'
qe.control.nstep                 = 1000

# SYSTEM
qe.system.ecutwfc                = 35.
qe.system.ecutrho                = 280.
qe.system.occupations            = 'smearing'
qe.system.smearing               = 'mv'
qe.system.degauss                = 0.001

# LDA+U & MAGNETIZATION
qe.lda_plus_u.u_dict             = {'Rh':3.5} # dictionary for U parameters
qe.magnet.mag_dict               = {'Rh':1.} # dictionary for magnetization

# ELECTRONS
qe.electrons.conv_thr            = 1.0e-6
qe.electrons.diagonalization     = 'david'
qe.electrons.diago_david_ndim    = 2
qe.electrons.mixing_beta         = 0.2
qe.electrons.mixing_mode         = 'local-TF'
qe.electrons.electron_maxstep    = 500

# IONS
qe.ions.ion_dynamics             = 'bfgs'

# CELL
qe.cell.cell_dynamics            = 'bfgs'
qe.cell.cell_dofree              = 'z'

# K POINTS
qe.kpoints.nk                    = k_points
qe.kpoints.type                  = 'automatic'

# ATOMIC POSITIONS
qe.atomic_positions.type         = 'angstrom'

################################################################################
#                               WRITE OUTPUTS
################################################################################

qe.atoms = system
qe.write_input('pw.inp')
system.write('pw.xsf')
system.write('pw.traj')

################################################################################
#                                    END
################################################################################
