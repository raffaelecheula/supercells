#!/usr/bin/python

################################################################################
# SUPERCELL BUILDER version 0.1, distributed under the GPLv3 license
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import division, print_function
import numpy as np
import copy as cp
from math import *
from ase import Atom, Atoms
from ase.io import write, read
from ase.build import *
from ase.units import *
from ase.constraints import FixAtoms

print('\n'
"###################################################################",'\n'
"# SUPERCELL BUILDER version 0.1",'\n'
"# Distributed under the GPLv3 license",'\n'
"# Author: Raffaele Cheula",'\n'
"# raffaele.cheula@polimi.it",'\n'
"# Laboratory of Catalysis and Catalytic Processes (LCCP)",'\n'
"# Department of Energy, Politecnico di Milano",'\n'
"###################################################################",'\n')

################################################################################
#                        INPUT COMMANDS DESCRIPTION
################################################################################

"""
bulk_type           = | 'fcc' | 'bcc' | 'hcp' | 'corundum' | 'rutile' |
                      | 'graphene' | [bulk_str, dim_cell, basis, elem_basis] |
input_bulk          = | None | (file.format, 'format', 'bulk_type') |

elements            = | 'M' | ('M', 'O') |
lattice_constants   = | a | (a, c) |

input_slab          = | None | (file.format, 'format') |
miller_index        = | '100' | '110' | '111' | '0001' | (h, k, l) |
surface_vectors     = | None | 'automatic' | [[xx, xy], [yx, yy]] |
dimensions          = | (x, y) |

layers              = | N layers |
layers_fixed        = | N layers fixed |
symmetry            = | None | 'planar' | 'inversion' |
cut_top             = | None | angstrom |
cut_bottom          = | None | angstrom |
vacuum              = | None | angstrom |

adsorbates          = | None | ('A', 'site', N ads, distance) | 
                      | [['A', x, y, distance], [mol, x, y, distance], ...] |
vacancies           = | None | [[x, y, z], ...] |
units               = | 'initial_cell' | 'final_cell' | 'angstrom' |
break_sym           = | True | False |
rotation_angle      = | None | 'automatic' |

k_points            = | None | [x, y, z] |
scale_kp            = | None | 'xy' | 'xyz' |
"""

################################################################################
#                                SUPERCELL
################################################################################

class Supercell:

    def __init__(self, **kwargs):
        self.bulk_type         = None
        self.input_bulk        = None
        self.elements          = None
        self.lattice_constants = None
        self.input_slab        = None
        self.miller_index      = None
        self.surface_vectors   = None
        self.dimensions        = [1, 1]
        self.layers            = 1
        self.layers_fixed      = None
        self.symmetry          = None
        self.cut_top           = None
        self.cut_bottom        = None
        self.vacuum            = None
        self.adsorbates        = None
        self.vacancies         = None
        self.units             = [[1., 0.], [0., 1.]]
        self.break_sym         = False
        self.sort_atoms        = False
        self.rotation_angle    = None
        self.k_points          = [1, 1, 1]
        self.scale_kp          = None
        self.epsi              = 1e-4
        self.dir_self = dir(self) + ['dir_self', 'kwargs', 'self']
        locals().update(kwargs)
        self.__dict__.update(locals())

    def create_system(self):
        for i in [ i for i in dir(self) if i not in self.dir_self ]:
            raise TypeError('got an unexpected keyword argument: %s' % i)
        bulk_type         = self.bulk_type
        input_bulk        = self.input_bulk
        elements          = self.elements
        lattice_constants = self.lattice_constants
        input_slab        = self.input_slab
        miller_index      = self.miller_index
        surface_vectors   = self.surface_vectors
        dimensions        = self.dimensions
        layers            = self.layers
        layers_fixed      = self.layers_fixed
        symmetry          = self.symmetry
        cut_top           = self.cut_top
        cut_bottom        = self.cut_bottom
        vacuum            = self.vacuum
        adsorbates        = self.adsorbates
        vacancies         = self.vacancies
        units             = self.units
        break_sym         = self.break_sym
        sort_atoms        = self.sort_atoms
        rotation_angle    = self.rotation_angle
        k_points          = self.k_points
        scale_kp          = self.scale_kp
        epsi              = self.epsi
        dim               = [1, 1]

#-------------------------------------------------------------------------------
#                         AUTOMATIC SURFACE VECTORS
#-------------------------------------------------------------------------------

        if surface_vectors is 'automatic':
            if input_slab is None:
                surface_vectors = automatic_surface_vectors(surface_vectors,
                                                            bulk_type,
                                                            miller_index)
            else:
                surface_vectors = None

#-------------------------------------------------------------------------------
#                                 INITIATION
#-------------------------------------------------------------------------------

        if type(elements) is str: 
            elements = [elements]
        if type(lattice_constants) is float: 
            lattice_constants = [lattice_constants]

        if surface_vectors:
            vector_a = (surface_vectors[0][0] * dimensions[0],
                        surface_vectors[0][1] * dimensions[0])
            vector_b = (surface_vectors[1][0] * dimensions[1],
                        surface_vectors[1][1] * dimensions[1])
        else:
            for i in range(len(dimensions)):
                dim[i] = int(dimensions[i])

#-------------------------------------------------------------------------------
#                                BULK SECTION
#-------------------------------------------------------------------------------

        if input_bulk is not None:
            system = read(input_bulk[0], format = input_bulk[1])

        elif bulk_type is 'simple_cubic':
            cell = [[lattice_constants[0], 0., 0.],
                    [0., lattice_constants[0], 0.], 
                    [0., 0., lattice_constants[0]]]
            system = Atoms(elements[0], scaled_positions = [[0., 0., 0.]],
                           cell = cell, pbc = True)

        elif bulk_type is 'fcc_reduced':
            a_lat = lattice_constants[0] / sqrt(2.)
            cell = [[a_lat, 0., 0.],
                    [a_lat / 2., a_lat * sqrt(3./4.), 0.],
                    [a_lat / 2., a_lat * sqrt(1./12.), a_lat * sqrt(2./3.)]]
            system = Atoms(elements[0], scaled_positions = [[0., 0., 0.]],
                           cell = cell, pbc = True)

        elif bulk_type in ('fcc', 'bcc'):
            system = bulk(elements[0], bulk_type, a = lattice_constants[0], 
                          cubic = True)

        elif bulk_type is 'hcp':
            system = bulk(elements[0], bulk_type, a = lattice_constants[0],
                          c = lattice_constants[1], cubic = True)

        else:
            system = custom_bulk(bulk_type = bulk_type, elements = elements, 
                                 lattice_constants = lattice_constants)

        if scale_kp is 'xy':
            unit_kp = np.array([[system.cell[0][0], system.cell[0][1]], 
                                [system.cell[1][0], system.cell[1][1]]])
        elif scale_kp is 'xyz':
            unit_kp = np.array(system.cell)

#-------------------------------------------------------------------------------
#                             SURFACE SECTION
#-------------------------------------------------------------------------------

        if miller_index is not None:
            system, height = create_surface(system, bulk_type = bulk_type, 
                                       miller_index = miller_index, dim = dim,
                                       elements = elements, layers = layers,
                                       lattice_constants = lattice_constants)
        else:
            height = system.cell[2][2]
            system *= (dim[0], dim[1], layers)

#-------------------------------------------------------------------------------
#                        INITIAL CELL UNITS SECTION
#-------------------------------------------------------------------------------

        unit = np.array([[system.cell[0][0] / dim[0],
                          system.cell[0][1] / dim[0]],
                         [system.cell[1][0] / dim[1],
                          system.cell[1][1] / dim[1]]])

        if miller_index in ('100', '110', '111', '0001'):
            if scale_kp is 'xy':
                unit_kp = np.array(unit)

#-------------------------------------------------------------------------------
#                            INPUT SLAB SECTION
#-------------------------------------------------------------------------------

        if input_slab:
            if input_slab[1] in ('espresso-out', 'out'):
                system = read_quantum_espresso_out(input_slab[0])
            else:
                system = read(input_slab[0], format = input_slab[1])
            system.center(vacuum = 0., axis = 2)
            system *= (dim[0], dim[1], 1)

#-------------------------------------------------------------------------------
#                          CUSTOM SURFACE SECTION
#-------------------------------------------------------------------------------

        if surface_vectors:
            big_dim = int(ceil(max(vector_a[0], vector_a[1], 
                                   vector_b[0], vector_b[1])) * 4)
            system = surface_cut(system, vector_a = vector_a, 
                                 vector_b = vector_b,
                                 big_dim = big_dim)

#-------------------------------------------------------------------------------
#                             ROTATION SECTION
#-------------------------------------------------------------------------------

        if rotation_angle is not None:
            if rotation_angle is 'automatic':
                rotation_angle = -atan(system.cell[0][1] / \
                                 system.cell[0][0]) * 180 / pi
                system.rotate(rotation_angle, v = 'z', rotate_cell = True)

            elif rotation_angle is 'invert_axis':
                rotation_angle = atan(system.cell[1][0] / \
                                 system.cell[1][1]) * 180 / pi + 90
                system.rotate(rotation_angle, v = 'z', rotate_cell = True)
                cell = np.array([[-system.cell[1][0], system.cell[1][1], 0.],
                                 [system.cell[0][0], system.cell[0][1], 0.],
                                 [0., 0., system.cell[2][2]]])
                system.translate((-system.cell[1][0], 0., 0.))
                system.set_cell(cell)
                system = surface_cut(system)
            else:
                system.rotate(rotation_angle, v = 'z', rotate_cell = True)

#-------------------------------------------------------------------------------
#                        CUT TOP AND BOTTOM SECTION
#-------------------------------------------------------------------------------

        if cut_top or cut_bottom:
            system = cut_supercell(system, cut_top = cut_top, 
                                   cut_bottom = cut_bottom, 
                                   vacuum = vacuum,
                                   epsi = epsi)

#-------------------------------------------------------------------------------
#                             SYMMETRY SECTION
#-------------------------------------------------------------------------------

        if break_sym is True and symmetry is None:
            trans = 1e-3
            for a in system:
                if a.position[2] > system.cell[2][2] - epsi:
                   a.position[2] = a.position[2] + trans

        if symmetry is 'inversion':
            system = surface_cut(system)
            system = inversion_symmetry(system)
            if vacuum is not None:
                system.center(vacuum = 0., axis = 2)

#-------------------------------------------------------------------------------
#                          FINAL CELL UNITS SECTION
#-------------------------------------------------------------------------------

        if units is 'final_cell':
            unit = np.array([[system.cell[0][0] / dim[0],
                              system.cell[0][1] / dim[0]],
                             [system.cell[1][0] / dim[1],
                              system.cell[1][1] / dim[1]]])

        if units is 'angstrom':
            unit = np.array([[1., 0.], [0., 1.]])

#-------------------------------------------------------------------------------
#                             VACANCIES SECTION
#-------------------------------------------------------------------------------

        if vacancies:
            system = create_vacancies(system, vacancies = vacancies, 
                                      symmetry = symmetry, 
                                      unit = unit)

#-------------------------------------------------------------------------------
#                            ADSORBATES SECTION
#-------------------------------------------------------------------------------

        if adsorbates:
            if len(adsorbates) is 4 and adsorbates[1] in ('top', 'hollow',
                'fcc', 'hcp', 'bridge', 'shortbridge', 'longbridge'):
                adsorbates = standard_adsorbates(adsorbates, 
                                                 bulk_type = bulk_type,
                                                 miller_index = miller_index)

            system = add_adsorbates(system, adsorbates = adsorbates,
                                    symmetry = symmetry, 
                                    unit = unit)

#-------------------------------------------------------------------------------
#                        SORT, FIX AND VACUUM SECTION
#-------------------------------------------------------------------------------

        if sort_atoms:
            args = np.argsort(system.positions[:, 2])
            system = system[args]

        if layers_fixed:
            system = fix_atoms(system, layers_fixed = layers_fixed, 
                               height = height, 
                               miller_index = miller_index, 
                               symmetry = symmetry)
    
        if vacuum is not None:
            system.center(vacuum = vacuum / 2., axis = 2)

        system.set_pbc((True, True, True))

#-------------------------------------------------------------------------------
#                             K POINTS SECTION
#-------------------------------------------------------------------------------

        if k_points is not None and scale_kp in ('xy', 'xyz'):
            k_points = scale_kpoints(system, k_points = k_points,
                                     scale_kp = scale_kp,
                                     unit_kp = unit_kp)

#-------------------------------------------------------------------------------
#                             RETURN SECTION
#-------------------------------------------------------------------------------

        self.height = height
        self.k_points = k_points

        return system

################################################################################
#                         AUTOMATIC SURFACE VECTORS
################################################################################

def automatic_surface_vectors(surface_vectors, bulk_type, miller_index):

    if bulk_type is 'fcc':
        if miller_index == (1, 0, 0):
            surface_vectors = [[0.5, 0.5], [-0.5, 0.5]]
        elif miller_index == (1, 1, 0):
            surface_vectors = [[0.5, 0.], [0., 1.]]
        elif miller_index == (1, 1, 1):
            surface_vectors = [[0.5, 0.], [0.5, 0.5]]
        elif miller_index == (2, 1, 0):
            surface_vectors = [[0.5, 0.5], [0., 1.]]
        elif miller_index == (2, 1, 1):
            surface_vectors = [[1., 0.], [0., 0.5]]
        elif miller_index == (2, 2, 1):
            surface_vectors = [[1., 0.], [-0.5, 0.5]]
        elif miller_index == (3, 1, 0):
            surface_vectors = [[0.5, 0.], [0., 1.]]
        elif miller_index == (3, 1, 1):
            surface_vectors = [[0.5, 0.], [0., 0.5]]
        elif miller_index == (3, 2, 0):
            surface_vectors = [[0.5, 0.5], [0., 1.]]
        elif miller_index == (3, 2, 1):
            surface_vectors = [[0.5, 0.], [-0.5, 1.]]
        elif miller_index == (3, 3, 1):
            surface_vectors = [[0.5, 0.], [-0.5, 0.5]]
        else:
            surface_vectors = None
    elif bulk_type is 'corundum':
        if miller_index == (0, 0, 1):
            surface_vectors = [[1., 0.], [0., 1.]]
        if miller_index == (1, -1, 2):
            surface_vectors = [[1., 0.], [-1./3., 1./3.]]
        else:
            surface_vectors = None
    else:
        surface_vectors = None # (to complete)

    return surface_vectors

################################################################################
#                              CREATE SURFACE
################################################################################

def create_surface(system, bulk_type, miller_index, dim, elements, layers,
                   lattice_constants):

    if miller_index is '100':
        if bulk_type is 'fcc':
            system = fcc100(elements[0], size = (dim[0], dim[1], layers),
                            a = lattice_constants[0], vacuum = 0.)
        elif bulk_type is 'bcc':
            system = bcc100(elements[0], size = (dim[0], dim[1], layers),
                            a = lattice_constants[0], vacuum = 0.)

    elif miller_index is '110':
        if bulk_type is 'fcc':
            system = fcc110(elements[0], size = (dim[0], dim[1], layers),
                            a = lattice_constants[0], vacuum = 0.)
        elif bulk_type is 'bcc':
            system = bcc110(elements[0], size = (dim[0], dim[1], layers),
                            a = lattice_constants[0], vacuum = 0.)

    elif miller_index is '111':
        if bulk_type is 'fcc':
            system = fcc111(elements[0], size = (dim[0], dim[1], layers),
                            a = lattice_constants[0], vacuum = 0.)
        elif bulk_type is 'bcc':
            system = bcc111(elements[0], size = (dim[0], dim[1], layers),
                            a = lattice_constants[0], vacuum = 0.)

    elif miller_index is '0001' and bulk_type is 'hcp':
        system = hcp0001(elements[0], size = (dim[0], dim[1], layers),
                         a = lattice_constants[0], 
                         c = lattice_constants[1],
                         vacuum = 0.)

    else:
        one_layer = surface(system, miller_index, 1, vacuum = 0.)
        height = one_layer.cell[2][2]
        if np.round(height, 3) == 0.:
            one_layer = surface(system, miller_index, 2, vacuum = 0.)
            height = one_layer.cell[2][2]
        system = surface(system, miller_index, int(layers))
        for a in system:
            a.position = - a.position
        system.translate((system.cell[0][0] + system.cell[1][0],
                         system.cell[0][1] + system.cell[1][1], 0.))
        system.center(vacuum = 0., axis = 2)
        system *= (dim[0], dim[1], 1)

    if miller_index in ('100', '110', '111', '0001'):
        if layers != 1:
            height = system.cell[2][2] / (layers - 1.)
        else:
            height = system.cell[2][2]

    return system, height

################################################################################
#                            CUT TOP CUT BOTTOM
################################################################################

def cut_supercell(system, cut_top, cut_bottom, vacuum, epsi):

    if vacuum is not None:
        system.center(vacuum = 0., axis = 2)

    if cut_top:
        print('deleted top atoms:', len([ a.index for a in system \
            if a.position[2] > cut_top + epsi ]))
        del system [[ a.index for a in system \
            if a.position[2] > cut_top + epsi ]]

    if cut_bottom:
        print('deleted bottom atoms:', len([ a.index for a in system \
            if a.position[2] < cut_bottom - epsi ]))
        del system [[ a.index for a in system \
            if a.position[2] < cut_bottom - epsi ]]

    if vacuum is not None:
        system.center(vacuum = 0., axis = 2)

    return system

################################################################################
#                               CUSTOM BULK
################################################################################

def custom_bulk(bulk_type, elements, lattice_constants):

    if bulk_type is 'corundum':
        dim_cell = (3, 3, 12)
        basis = bravais = [[1., 2., 0.], [2., 1., 0.], [0., 2., 1.],
                           [1., 1., 1.], [2., 0., 1.], [0., 0., 2.],
                           [2., 1., 2.], [0., 1., 3.], [1., 0., 3.],
                           [2., 2., 3.], [0., 0., 4.], [1., 2., 4.],
                           [0., 2., 5.], [1., 1., 5.], [2., 0., 5.],
                           [1., 2., 6.], [2., 1., 6.], [0., 1., 7.],
                           [1., 0., 7.], [2., 2., 7.], [0., 0., 8.],
                           [2., 1., 8.], [0., 2., 9.], [1., 1., 9.],
                           [2., 0., 9.], [0., 0., 10.], [1., 2., 10.],
                           [0., 1., 11.], [1., 0., 11.], [2., 2., 11.]]
        elem_basis = (0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                      1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1)
        bulk_str = 'hexagonal'

    elif bulk_type is 'rutile':
        dim_cell = (6, 6, 2)
        basis = bravais = [[0., 0., 0.], [1.85, 4.15, 0.],
                           [4.15, 1.85, 0.], [1.15, 1.15, 1.],
                           [3., 3., 1.], [4.85, 4.85, 1.]]
        elem_basis = (0, 1, 1, 1, 0, 1)
        bulk_str = 'tetragonal'

    elif bulk_type is 'graphene':
        dim_cell = (3, 3, 2)
        basis = bravais = [[0., 0., 0.], [1., 2., 0.], [0., 0., 1.],
                           [2., 1., 1.]]
        elem_basis = (0, 0, 0, 0)
        bulk_str = 'hexagonal'

    else:
        bulk_str = bulk_type[0]
        dim_cell = bulk_type[1]
        basis = bravais = bulk_type[2]
        elem_basis = bulk_type[3]

    for i in range(len(basis)):
        for j in range(3):
            bravais[i][j] = basis[i][j] / float(dim_cell[j])

    if bulk_str is 'cubic':
        from ase.lattice.cubic import SimpleCubicFactory
        class CustomCellFactory(SimpleCubicFactory):
            bravais_basis = bravais
            element_basis = elem_basis
    elif bulk_str is 'hexagonal':
        from ase.lattice.hexagonal import HexagonalFactory
        class CustomCellFactory(HexagonalFactory):
            bravais_basis = bravais
            element_basis = elem_basis
    elif bulk_str is 'tetragonal':
        from ase.lattice.triclinic import TriclinicFactory
        class CustomCellFactory(SimpleTetragonalFactory):
            bravais_basis = bravais
            element_basis = elem_basis
    elif bulk_str is 'triclinic':
        from ase.lattice.tetragonal import SimpleTetragonalFactory
        class CustomCellFactory(TriclinicFactory):
            bravais_basis = bravais
            element_basis = elem_basis

    CustomCell = CustomCellFactory()
    system = CustomCell(symbol = elements[:max(elem_basis) + 1],
                        latticeconstant = lattice_constants,
                        size = (1, 1, 1))

    return system

################################################################################
#                               SURFACE CUT
################################################################################

def surface_cut(system, vector_a = [1., 0.], vector_b = [0., 1.],
                big_dim = 10, origin = [0., 0.], epsi = 1e-5):

    unit = np.array([[system.cell[0][0], system.cell[0][1]], 
                     [system.cell[1][0], system.cell[1][1]]])

    system *= (big_dim, big_dim, 1)
    system.translate((- big_dim / 2. * (unit[0][0] + unit[1][0]) - origin[0],
                      - big_dim / 2. * (unit[0][1] + unit[1][1]) - origin[1],
                      0.))

    cell = np.array([[vector_a[0] * unit[0][0] + vector_a[1] * unit[1][0],
                      vector_a[1] * unit[1][1]],
                     [vector_b[0] * unit[0][0] + vector_b[1] * unit[1][0],
                      vector_b[1] * unit[1][1]]])

    del system [[ a.index for a in system \
        if a.position[1] < cell[0][1] / cell[0][0] * a.position[0] - epsi
        or a.position[1] > cell[0][1] / cell[0][0] * \
                           (a.position[0] - cell[1][0]) + cell[1][1] - epsi
        or a.position[0] < cell[1][0] / cell[1][1] * a.position[1] - epsi
        or a.position[0] > cell[1][0] / cell[1][1] * \
                           (a.position[1] - cell[0][1]) + cell[0][0] - epsi ]]

    system.set_cell(np.matrix([[cell[0][0], cell[0][1], 0.],
                               [cell[1][0], cell[1][1], 0.],
                               [0., 0., system.cell[2][2]]]))

    return system

################################################################################
#                             BOUNDARY ATOMS
################################################################################

def boundary_atoms(system, base_boundary = False, outer_boundary = False, 
                   epsi = 1e-4):

    system_plus = cp.deepcopy(system)

    for a in system:
        if abs(a.position[0]) < epsi and abs(a.position[1]) < epsi:
            a_plus = cp.deepcopy(a)
            a_plus.position[0] += (system.cell[0][0] + system.cell[1][0])
            a_plus.position[1] += (system.cell[0][1] + system.cell[1][1])
            system_plus += a_plus

        if abs(a.position[0] - a.position[1] * system.cell[1][0] / \
           system.cell[1][1]) < epsi:
            a_plus = cp.deepcopy(a)
            a_plus.position[0] += system.cell[0][0]
            a_plus.position[1] += system.cell[0][1]
            system_plus += a_plus

        if abs(a.position[1] - a.position[0] * system.cell[0][1] / \
           system.cell[0][0]) < epsi:
            a_plus = cp.deepcopy(a)
            a_plus.position[0] += system.cell[1][0]
            a_plus.position[1] += system.cell[1][1]
            system_plus += a_plus

    if base_boundary is True:
        for a in system_plus:
            if abs(a.position[2]) < epsi:
                a_plus = cp.deepcopy(a)
                a_plus.position[2] += system.cell[2][2]
                system_plus += a_plus

    if outer_boundary is True:
        for a in system_plus:
            if abs(a.position[0] - system.cell[0][0] - system.cell[1][0]) < \
               epsi and abs(a.position[1] - system.cell[0][1] - \
               system.cell[1][1]) < epsi:
                a_plus = cp.deepcopy(a)
                a_plus.position[0] -= (system.cell[0][0] + system.cell[1][0])
                a_plus.position[1] -= (system.cell[0][1] + system.cell[1][1])
                system_plus += a_plus

            if abs(a.position[0] - system.cell[0][0] - a.position[1] * \
               system.cell[1][0] / system.cell[1][1]) < epsi:
                a_plus = cp.deepcopy(a)
                a_plus.position[0] -= system.cell[0][0]
                a_plus.position[1] -= system.cell[0][1]
                system_plus += a_plus

            if abs(a.position[1] - system.cell[1][1] - a.position[0] * \
               system.cell[0][1] / system.cell[0][0]) < epsi:
                a_plus = cp.deepcopy(a)
                a_plus.position[0] -= system.cell[1][0]
                a_plus.position[1] -= system.cell[1][1]
                system_plus += a_plus

    return system_plus

################################################################################
#                          CHECK INVERSION SYMMETRY
################################################################################

def check_inversion_symmetry(system, base_boundary = False, 
                             outer_boundary = False, print_check = False):

    print('')
    print('CHECKING INVERSION SYMMETRY')
    cont = 0
    system = boundary_atoms(system, base_boundary = base_boundary,
                            outer_boundary = outer_boundary)
    system.center(vacuum = 0., axis = 2)
    centre = (system.cell[0] + system.cell[1] + system.cell[2]) / 2.

    for a in system:
        a_check = 2. * centre - a.position
        equal = False
        equal_check = False
        for b in system:
            equal = np.allclose(a_check, b.position, 
                                rtol = 1e-02, atol = 1e-03)
            if equal is True:
                cont += 1
                equal_check = True
                break
        if equal_check is False and print_check is True:
            print(np.around(a.position, decimals = 3), 
                  'has no corrispondent in:',
                  np.around(a_check, decimals = 3))

    print('check inversion:', len(system), cont)

    if cont >= len(system):
        return True
    else:
        return False

################################################################################
#                         CREATE INVERSION SYMMETRY
################################################################################

def create_inversion_symmetry(system, base_boundary = False, 
                              outer_boundary = False, epsi = 1e-5):

    print('')
    print('CREATING INVERSION SYMMETRY')
    system_plus = boundary_atoms(system, base_boundary = False,
                                 outer_boundary = False)

    c_matrix = np.array([ ((a.position + b.position) / 2.) for a in \
        system_plus for b in system_plus if b.symbol == a.symbol and \
        b.index >= a.index ])

    print('number of centres =', len(c_matrix))

    indices = np.array([ len([ j for j in range(i, len(c_matrix)) if \
            np.array_equal(np.around(c_matrix[i], decimals = 3),
            np.around(c_matrix[j], decimals = 3)) ]) for i in \
            range(len(c_matrix)) ])

    print('number of occurrences =', max(indices))

    centre = c_matrix[np.argmax(indices)]
    origin = centre - (system.cell[0] + system.cell[1] + system.cell[2]) / 2.

    if origin[2] < 0.:
        print('deleted top atoms:', len([ a.index for a in system \
            if a.position[2] > 2. * centre[2] + epsi ]))
        del system [[ a.index for a in system \
            if a.position[2] > 2. * centre[2] + epsi ]]
    elif origin[2] > 0.:
        print('deleted bottom atoms:', len([ a.index for a in system \
            if a.position[2] < 2. * centre[2] - system.cell[2][2] - epsi ]))
        del system [[ a.index for a in system \
            if a.position[2] < 2. * centre[2] - system.cell[2][2] - epsi ]]

    system = surface_cut(system, origin = [origin[0], origin[1]])

    return system

################################################################################
#                             INVERSION SYMMETRY
################################################################################

def inversion_symmetry(system, big_cell_inversion = False):

    sym = check_inversion_symmetry(system, base_boundary = True,
                                   outer_boundary = True, print_check = False)
    print('inversion symmetry =', sym)

    if sym is not True:
        system_inv = create_inversion_symmetry(system)
        sym = check_inversion_symmetry(system_inv, print_check = True)
        print('inversion symmetry =', sym)

        if sym is not True and big_cell_inversion is True:
            print('BIG CELL INVERSION SYMMETRY')
            origin = [(system.cell[0][0] + system.cell[1][0]) / 2.,
                      (system.cell[0][1] + system.cell[1][1]) / 2.]
            system *= (2, 2, 1)
            system = create_inversion_symmetry(system)
            system = surface_cut(system, vector_a = [0.5, 0.],
                                 vector_b = [0., 0.5], big_dim = 2,
                                 origin = origin)
            sym = check_inversion_symmetry(system)
            print('inversion symmetry =', sym)
            if sym is not True:
                print('NO SYMMETRY FOUND!')
        else:
            system = system_inv

    return system

################################################################################
#                          STANDARD ADSORBATES
################################################################################

def standard_adsorbates(adsorbates, bulk_type = None, miller_index = None):

    ads = [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
           [[0.5, 0.0], [1.5, 0.0], [0.5, 1.0], [1.5, 1.0]],
           [[0.0, 0.5], [0.0, 1.5], [1.0, 0.5], [1.0, 1.5]],
           [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]],
           [[1./3., 1./3.], [4./3., 1./3.], [1./3., 4./3.], [1./3., 4./3.]],
           [[2./3., 2./3.], [5./3., 2./3.], [2./3., 5./3.], [2./3., 5./3.]]]

    if bulk_type is 'bcc' and miller_index is '110':
        ads[3] = [[1./3., 1./3.], [4./3., 1./3.], [1./3., 4./3.],
                  [4./3., 4./3.]]

    if miller_index == (2, 1, 1):
        ads = [[[0., 0.], [], [], []],
               [[0.2, 0.2], [], [], []],
               [[0., 0.25], [], [], []],
               [[0.2, 0.45], [], [], []],
               [[], [], [], []],
               [[], [], [], []]]

    if miller_index == (3, 1, 1):
        ads = [[[0., 0.], [], [], []],
               [[1./3., 0.25], [], [], []],
               [[0., 0.25], [], [], []],
               [[1./3., 0.5], [], [], []],
               [[], [], [], []],
               [[], [], [], []]]

    if miller_index == (3, 3, 1):
        ads = [[[0., 0.], [], [], []],
               [[0.38, 0.12], [], [], []],
               [[0., 0.25], [], [], []],
               [[0.38, 0.35], [], [], []],
               [[], [], [], []],
               [[], [], [], []]]

    if miller_index in ('111', '0001') and adsorbates[1] is 'hollow':
        adsorbates[1] = 'fcc'

    site = {'top':0, 'bridge':1, 'longbridge':1, 'shortbridge':2,
            'hollow':3, 'fcc':4, 'hcp':5}

    num = {1:[0], 2:[0, 3], 3:[0, 1, 2], 4:[0, 1, 2, 3]}

    j = 0
    ads_vector = [[0] for i in range(adsorbates[2])]
    for i in num[adsorbates[2]]:
        ads_vector[j] = [adsorbates[0], ads[site[adsorbates[1]]][i][0],
                         ads[site[adsorbates[1]]][i][1], adsorbates[3]]
        j += 1

    return ads_vector

################################################################################
#                             ADD ADSORBATES
################################################################################

def add_adsorbates(system, adsorbates, symmetry = None, all_scaled = False,
                   unit = [[1., 0.], [0., 1.]]):

    for i in range(len(adsorbates)):
        if isinstance(adsorbates[i][0], (list, tuple)) is False:
            ads_pos = (adsorbates[i][1] * unit[0][0] + \
                       adsorbates[i][2] * unit[1][0], 
                       adsorbates[i][2] * unit[1][1] + \
                       adsorbates[i][1] * unit[0][1])
            adsorbate = Atom(adsorbates[i][0], (ads_pos[0], ads_pos[1],
                             system.cell[2][2] + adsorbates[i][3]))

            system += adsorbate

            if symmetry is 'planar':
                adsorbate = Atom(adsorbates[i][0], (ads_pos[0], ads_pos[1],
                                 - adsorbates[i][3]))
                system += adsorbate

            elif symmetry is 'inversion':
                ads_pos = (system.cell[0][0] + system.cell[1][0] - ads_pos[0],
                           system.cell[1][1] + system.cell[0][1] - ads_pos[1])
                adsorbate = Atom(adsorbates[i][0], (ads_pos[0], ads_pos[1],
                                 - adsorbates[i][3]))
                system += adsorbate

        else:
            for j in range(len(adsorbates[i][0])):
                if all_scaled is True:
                    ads_pos = ((adsorbates[i][1] + adsorbates[i][0][j][1]) * \
                               unit[0][0] + (adsorbates[i][2] + \
                               adsorbates[i][0][j][2]) * unit[1][0],
                               (adsorbates[i][2] + adsorbates[i][0][j][2]) * \
                               unit[1][1] + (adsorbates[i][1] + \
                               adsorbates[i][0][j][1]) * unit[0][1])
                else:
                    ads_pos = (adsorbates[i][1] * unit[0][0] + \
                               adsorbates[i][0][j][1] + \
                               adsorbates[i][2] * unit[1][0],
                               adsorbates[i][2] * unit[1][1] + \
                               adsorbates[i][0][j][2] + \
                               adsorbates[i][1] * unit[0][1])
                adsorbate = Atom(adsorbates[i][0][j][0], (ads_pos[0],
                                 ads_pos[1], system.cell[2][2] + \
                                 adsorbates[i][3] + adsorbates[i][0][j][3]))

                system += adsorbate

                if symmetry is 'planar':
                    adsorbate = Atom(adsorbates[i][0][j][0], (ads_pos[0],
                                     ads_pos[1], - adsorbates[i][3] - \
                                     adsorbates[i][0][j][3]))

                    system += adsorbate

                elif symmetry is 'inversion':
                    ads_pos = (system.cell[0][0] + system.cell[1][0] - \
                               ads_pos[0], system.cell[1][1] + \
                               system.cell[0][1] - ads_pos[1])
                    adsorbate = Atom(adsorbates[i][0][j][0], (ads_pos[0],
                                     ads_pos[1], - adsorbates[i][3] - \
                                     adsorbates[i][0][j][3]))

                    system += adsorbate

    return system

################################################################################
#                             CREATE VACANCIES
################################################################################

def create_vacancies(system, vacancies, symmetry = None,
                     unit = [[1., 0.], [0., 1.]], epsi = 1e-4):

    for i in range(len(vacancies)):
        vacancy = (vacancies[i][0] * unit[0][0] + \
                   vacancies[i][1] * unit[1][0], 
                   vacancies[i][1] * unit[1][1] + \
                   vacancies[i][0] * unit[0][1], 
                   system.cell[2][2] + vacancies[i][2])

        del system [[ a.index for a in system \
            if vacancy[0] - epsi <= a.position[0] <= vacancy[0] + epsi
            and vacancy[1] - epsi <= a.position[1] <= vacancy[1] + epsi
            and vacancy[2] - epsi <= a.position[2] <= vacancy[2] + epsi ]]

        if symmetry is 'planar':
            vacancy[2] = - vacancies[i][2]
            del system [[ a.index for a in system \
                if vacancy[0] - epsi <= a.position[0] <= vacancy[0] + epsi
                and vacancy[1] - epsi <= a.position[1] <= vacancy[1] + epsi
                and vacancy[2] - epsi <= a.position[2] <= vacancy[2] + epsi ]]

        elif symmetry is 'inversion':
            vacancy = (system.cell[0][0] + system.cell[1][0] - vacancy[0],
                       system.cell[1][1] + system.cell[0][1] - vacancy[1],
                       - vacancies[i][2])
            del system [[ a.index for a in system \
                if vacancy[0] - epsi <= a.position[0] <= vacancy[0] + epsi
                and vacancy[1] - epsi <= a.position[1] <= vacancy[1] + epsi
                and vacancy[2] - epsi <= a.position[2] <= vacancy[2] + epsi ]]

    return system

################################################################################
#                                 FIX ATOMS
################################################################################

def fix_atoms(system, layers_fixed, height, miller_index, symmetry = None, 
              epsi = 1e-4):

    if symmetry is None:
        indices = [a.index for a in system \
                   if a.position[2] < layers_fixed * height - epsi]

    elif symmetry in ('planar', 'inversion'):
        if miller_index in ('100', '110', '111', '0001') and \
           layers_fixed % 2 is not 0.:
            indices = [a.index for a in system if 0.5 * system.cell[2][2] - \
                       (layers_fixed - 1.) / 2. * height - epsi < \
                       a.position[2] < 0.5 * system.cell[2][2] + \
                       (layers_fixed - 1.) / 2. * height + epsi]

        else:
            indices = [a.index for a in system if 0.5 * system.cell[2][2] - \
                       layers_fixed / 2. * height - epsi < \
                       a.position[2] < 0.5 * system.cell[2][2] + \
                       layers_fixed / 2. * height + epsi]

    system.set_constraint(FixAtoms(indices = indices))

    return system

################################################################################
#                              SCALE K POINTS
################################################################################

def scale_kpoints(system, k_points, unit_kp, scale_kp, epsi = 1e-4):

    k_points[0] = sqrt(unit_kp[0][0] ** 2. + unit_kp[0][1] ** 2.) / \
                  sqrt(system.cell[0][0] ** 2. + system.cell[0][1] ** 2.) * \
                  k_points[0] - epsi
    k_points[1] = sqrt(unit_kp[1][0] ** 2. + unit_kp[1][1] ** 2.) / \
                  sqrt(system.cell[1][0] ** 2. + system.cell[1][1] ** 2.) * \
                  k_points[1] - epsi
    k_points[0] = int(np.ceil(k_points[0]))
    k_points[1] = int(np.ceil(k_points[1]))

    if scale_kp is 'xyz':
        k_points[2] = unit_kp[2][2] / system.cell[2][2] * k_points[2] - epsi
        k_points[2] = int(np.ceil(k_points[2]))

    return k_points

################################################################################
#                            WULFF CONSTRUCTION
################################################################################

def Wulff_construction(system, size, diameter, surface_energy_dict,
                       adhesion_energy_dict = None,
                       fill_the_vacuum = False,
                       vacuum_symbol = 'X',
                       fill_the_support = True,
                       support_symbol = 'S'):

    cell_vectors = system.cell
    system *= size
    system.translate(( - system.cell[0] - system.cell[1] - \
                      system.cell[2]) / 2.)

    miller_index = surface_energy_dict.keys()
    if adhesion_energy_dict is not None:
        support_index = adhesion_energy_dict.keys()

    index_matrix = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                    [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                    [[0, 0, 1], [0, 1, 0], [1, 0, 0]]]

    for i in range(len(miller_index)):
        index_vect = miller_index[i] / np.dot(miller_index[i], (1, 1, 1))**0.5
        for j in range(6):
            index_vector = np.dot(index_matrix[j], index_vect)
            if fill_the_vacuum is False:
                del system [[a.index for a in system \
                    if np.dot(abs(a.position), index_vector) > \
                        diameter / 2 * surface_energy_dict[miller_index[i]]]]
            else:
                for a in system: 
                    if np.dot(abs(a.position), index_vector) > \
                        diameter / 2 * surface_energy_dict[miller_index[i]]:
                        a.symbol = vacuum_symbol

    if adhesion_energy_dict is not None:
        index_vector = support_index / np.dot(support_index, (1, 1, 1))**0.5
        index_vector = index_vector[0]
        if fill_the_support is False:
            del system [[a.index for a in system \
                if np.dot(a.position, index_vector) > \
                    diameter / 2 * adhesion_energy_dict[support_index[0]]]]
        else:
            for a in system: 
                if np.dot(a.position, index_vector) > \
                    diameter / 2 * adhesion_energy_dict[support_index[0]]:
                    a.symbol = support_symbol

    system.translate((system.cell[0] + system.cell[1] + system.cell[2]) / 2.)

    return system

################################################################################
#                           CREATE INTERFACE SLAB
################################################################################

def create_interface_slab(support, system, inter_dist):

    support_height = support.cell[2][2]
    system_height = system.cell[2][2]

    support.translate([0., 0., system_height / 2. + inter_dist])
    system.translate([0., 0., - system_height / 2.])

    for a in system:
        if a.position[2] < 0.:
            a.position[2] += system_height + support_height + 2. * inter_dist

    print('')
    print('CREATING INTERFACE')
    print('initial cell base =',
          '[', '%9.5f %9.5f' %(system.cell[0][0], system.cell[0][1]), ']')
    print('                   ',
          '[', '%9.5f %9.5f' %(system.cell[1][0], system.cell[1][1]), ']')
    print('final cell base =  ',
          '[', '%9.5f %9.5f' %(support.cell[0][0], support.cell[0][1]), ']')
    print('                   ',
          '[', '%9.5f %9.5f' %(support.cell[1][0], support.cell[1][1]), ']')

    system.set_cell([support.cell[0], support.cell[1], system.cell[2]],
                    scale_atoms = True)

    support += system
    system = support

    base = False
    for a in system:
        if a.position[2] < 0.01:
            base = True

    system.center(vacuum = 0., axis = 2)
    if base is False:
        system.center(vacuum = (system_height + support_height + 2. * \
                      inter_dist  - system.cell[2][2]) / 2., axis = 2)

    return system

################################################################################
#                                    END
################################################################################
