################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import re
import numpy as np
import copy as cp
from math import pi, atan, ceil
from functools import reduce
from fractions import gcd
from collections import OrderedDict
from ase import Atom, Atoms
from supercell_builder import (cut_surface, rotate_slab, convert_miller_index,
                               check_inversion_symmetry)

################################################################################
# GET ATOM LIST
################################################################################

def get_atom_list(atoms):

    symbols = atoms.get_chemical_symbols()
    magmoms = [a.magmom for a in atoms]
    if len(symbols) > 1:
        for i in range(len(symbols)-1, 0, -1):
            for j in range(i):
                if symbols[j] == symbols[i] and magmoms[i] == magmoms[j]:
                    del symbols[i]
                    break

    return symbols

################################################################################
# GET ATOM DICT
################################################################################

def get_atom_dict(atoms):

    atom_dict = {}
    symbols = atoms.get_chemical_symbols()
    for symbol in symbols:
        if symbol in atom_dict:
            atom_dict[symbol] += 1
        else:
            atom_dict[symbol] = 1

    return atom_dict

################################################################################
# GET FORMULA UNITS
################################################################################

def get_formula_units(atoms):

    formula = atoms.get_chemical_formula(mode = 'hill')
    numbers = re.split('[A-Z]|[a-z]', formula)
    numbers = [ int(n) for n in numbers if n != '']
    formula_units = reduce(gcd, numbers)

    return formula_units

################################################################################
# ARRAY FROM DICT
################################################################################

def array_from_dict(symbol, array_dict):

    array = [] * len(symbol)
    for i in range(len(symbol)):
        if symbol[i] in array_dict:
            array[i] = array_dict[symbol[i]]

    return array

################################################################################
# READ VIB ENERGIES
################################################################################

def read_vib_energies(filename = 'log.txt', imaginary = False):

    vib_energies = []
    fileobj = open(filename, 'rU')
    lines = fileobj.readlines()
    fileobj.close()

    for i in range(3, len(lines)-2):
        string = lines[i].split()[1]
        if string[-1] == 'i':
            if imaginary is True:
                vib_energies.append(complex(0., float(string[:-1]) * 1e-3))
        else:
            vib_energies.append(complex(float(string) * 1e-3))

    return vib_energies

################################################################################
# ATOMS FIXED
################################################################################

def atoms_fixed(atoms):

    fixed = np.concatenate([a.__dict__['index'] for a in atoms.constraints if \
                            a.__class__.__name__ == 'FixAtoms'])

    return fixed

################################################################################
# ATOMS NOT FIXED
################################################################################

def atoms_not_fixed(atoms):

    fixed = atoms_fixed(atoms)
    not_fixed = [i for i in range(len(atoms)) if i not in fixed]
    
    return not_fixed

################################################################################
# MERGE SUPERCELLS
################################################################################

def merge_supercells(one, two, epsi = 5e-1, vector = (0., 0.)):

    atoms = one[:]
    
    origin = (two.cell[0][:2] * vector[0] + two.cell[1][:2] * vector[1])
    two = cut_surface(two, origin = origin)

    for b in two:
        new_atom = True
        for a in one:
            for b_position in [ b.position,
                                b.position + sum(atoms.cell[:2]),
                                b.position - sum(atoms.cell[:2]),
                                b.position + atoms.cell[0],
                                b.position - atoms.cell[0],
                                b.position + atoms.cell[1],
                                b.position - atoms.cell[1] ]:
                if np.allclose(a.position, b_position, atol = epsi):
                    a.position = (a.position + b_position)/2.
                    new_atom = False
        if new_atom:
            atoms.append(b)

    return atoms

################################################################################
# POTENTIAL ENERGY SAMPLING
################################################################################

def potential_energy_sampling_grid(slab, adsorbate):

    if slab.miller_index == '100':
        pos = [[+0.0, +0.0], [+1/6, +1/6], [+1/3, +1/3], [+1/4, +0.0],
               [+1/2, +0.0], [+1/2, +1/4], [+1/2, +1/2]]

    elif slab.miller_index == '110':
        pos = [[+0.0, +0.0], [+1/6, +1/6], [+1/3, +1/3], [+1/4, +0.0],
               [+1/2, +0.0], [+1/2, +1/4], [+1/2, +1/2], [+0.0, +1/2],
               [+1/4, +1/2]]

    elif slab.miller_index == '111':
        pos = [[+0.0, +0.0], [+1/6, +1/6], [+1/3, +1/3], [+1/4, +0.0],
               [+1/2, +0.0], [+2/6, -1/6], [+2/3, -1/3]]
    
    elif slab.miller_index == '211':
        pos = [[+0.0, +0.0], [+1/6, +0.0], [+1/3, +0.0], [+1/2, +0.0],
               [+2/3, +0.0], [+5/6, +0.0], [+0.0, +1/4], [+1/6, +1/4],
               [+1/3, +1/4], [+1/2, +1/4], [+2/3, +1/4], [+5/6, +1/4],
               [+0.0, +1/2], [+1/6, +1/2], [+1/3, +1/2], [+1/2, +1/2],
               [+2/3, +1/2], [+5/6, +1/2]]

    positions_dict = {}
    for i in range(len(pos)):
        positions_dict['{0:.6f}_{1:.6f}'.format(pos[i][0], pos[i][1])] = pos[i]

    return positions_dict

################################################################################
# CREATE INTERFACE SLAB
################################################################################

def create_interface_slab(slab_a, slab_b, distance = 1.5,
                          symmetry = 'inversion', adapt_dimensions = True,
                          area_max = 50., nmax = None, stretch_max = 2.,
                          toll_rotation = 1e-3):

    slab_a = slab_a[:]
    slab_b = slab_b[:]

    if adapt_dimensions:
        slab_a, slab_b = adapt_slabs_dimensions(slab_a, slab_b,
                                                area_max      = area_max,
                                                nmax          = nmax,
                                                stretch_max   = stretch_max,
                                                toll_rotation = toll_rotation)

    slab_a.center(vacuum = 0., axis = 2)
    slab_b.center(vacuum = 0., axis = 2)

    height_a = slab_a.cell[2][2]
    height_b = slab_b.cell[2][2]

    slab_b.set_cell(np.vstack([slab_a.cell[:2], slab_b.cell[2]]),
                    scale_atoms = True)

    if symmetry in ('planar', 'inversion'):
        slab_a.translate([0., 0., +height_b/2. + distance])
        slab_b.translate([0., 0., -height_b/2.])
        for a in [ a for a in slab_b if a.position[2] < -1e-5 ]:
            a.position[2] += height_a + height_b + 2*distance
        atoms = slab_a + slab_b
        atoms.cell[2][2] = height_a + height_b + 2*distance
        sym = check_inversion_symmetry(atoms, 
                                       base_boundary  = True, 
                                       outer_boundary = True)
        print('inversion symmetry =', sym, '\n')
    else:
        slab_b.translate([0., 0., height_a + distance])
        atoms = slab_a[:]
        atoms += slab_b
        atoms.cell[2][2] = height_a + height_b + 2*distance

    return atoms

################################################################################
# ADAPT SLABS DIMENSIONS
################################################################################

def adapt_slabs_dimensions(slab_a, slab_b, area_max = 100., stretch_max = 0.05,
                           toll_rotation = 1e-3, epsi = 1e-4, nmax = None):

    cell_a = np.vstack([slab_a.cell[:2][0][:2], slab_a.cell[:2][1][:2]])
    cell_b = np.vstack([slab_b.cell[:2][0][:2], slab_b.cell[:2][1][:2]])

    if nmax is None:
        area_cell_a = cell_a[0][0]*cell_a[1][1] - cell_a[0][1]*cell_a[1][0]
        area_cell_b = cell_b[0][0]*cell_b[1][1] - cell_b[0][1]*cell_b[1][0]
        nmax_a = int(ceil(area_max / area_cell_a))
        nmax_b = int(ceil(area_max / area_cell_b))
    else:
        nmax_a = nmax_b = nmax

    try:
        print('USING CYTHON')
        from supercell_cython import match_slabs_dimensions_cython
        vect_a_opt, vect_b_opt, invert_opt, match_dimensions = \
            match_slabs_dimensions_cython(cell_a, cell_b, nmax_a, nmax_b,
                                          stretch_max, area_max,
                                          toll_rotation, epsi)
    except ImportError:
        vect_a_opt, vect_b_opt, invert_opt, match_dimensions = \
            match_slabs_dimensions_python(cell_a, cell_b, nmax_a, nmax_b,
                                          stretch_max, area_max,
                                          toll_rotation, epsi)

    if match_dimensions is False:
        print('NO MATCH BETWEEN THE SLABS IS FOUND!')
        return slab_a, slab_b

    slab_a = cut_surface(slab_a, surface_vectors = vect_a_opt)
    slab_a = rotate_slab(slab_a, 'automatic')
    slab_b = cut_surface(slab_b, surface_vectors = vect_b_opt)
    slab_b = rotate_slab(slab_b, 'automatic')
    
    if invert_opt is True:
        slab_b = rotate_slab(slab_b, 'invert_axis')

    print('FINAL STRUCTURE')
    print('surface vectors a =', vect_a_opt)
    print('surface vectors b =', vect_b_opt)
    print('invert axis b =', invert_opt, '\n')

    print('base slab a = [[{0:8.4f} {1:8.4f}]'.format(slab_a.cell[0][0],
                                                      slab_a.cell[0][1]))
    print('               [{0:8.4f} {1:8.4f}]]\n'.format(slab_a.cell[1][0],
                                                         slab_a.cell[1][1]))
    print('base slab b = [[{0:8.4f} {1:8.4f}]'.format(slab_b.cell[0][0],
                                                      slab_b.cell[0][1]))
    print('               [{0:8.4f} {1:8.4f}]]\n'.format(slab_b.cell[1][0],
                                                         slab_b.cell[1][1]))

    return slab_a, slab_b

################################################################################
# MATCH SLABS DIMENSIONS PYTHON
################################################################################

def match_slabs_dimensions_python(cell_a, cell_b, nmax_a, nmax_b, stretch_max,
    area_max, toll_rotation, epsi, minimum_angle = pi/5):

    iter_a_00_11 = [ i for i in range(1, nmax_a+1) ]

    index_a = [0]
    for i in range(1, nmax_a): index_a += [i, -i]
    iter_a_01_10 = [ i for i in index_a ]

    iter_b_00_11 = [ i for i in range(1, nmax_b+1) ]

    index_b = [0]
    for i in range(1, nmax_b): index_b += [i, -i]
    iter_b_01_10 = [ i for i in index_b ]

    vect_a = [[1, 0], [0, 1]]
    vect_b = [[1, 0], [0, 1]]
    invert = False

    vect_a_opt = np.copy(vect_a)
    vect_b_opt = np.copy(vect_b)
    invert_opt = False

    print('ADAPTING SLABS DIMENSIONS\n')
    print('nmax slab a = {}'.format(nmax_a))
    print('nmax slab b = {}\n'.format(nmax_b))

    deform_nrgs = []
    areas = []
    match_dimensions = False

    for a00, a01, a10, a11 in [ (a00, a01, a10, a11) \
                                for a01 in iter_a_01_10 \
                                for a10 in iter_a_01_10 \
                                for a00 in iter_a_00_11 \
                                for a11 in iter_a_00_11 ]:

        new_a = np.array([[cell_a[0][0]*a00 + cell_a[1][0]*a01,
                           cell_a[0][1]*a00 + cell_a[1][1]*a01],
                          [cell_a[0][0]*a10 + cell_a[1][0]*a11,
                           cell_a[0][1]*a10 + cell_a[1][1]*a11]])

        area = new_a[0][0]*new_a[1][1] - new_a[0][1]*new_a[1][0]

        if new_a[0][0] > epsi and new_a[1][1] > epsi and area < area_max:

            angle_a = pi/2 - atan(new_a[0][1]/new_a[0][0]) \
                           - atan(new_a[1][0]/new_a[1][1])

            if minimum_angle < angle_a < pi - minimum_angle:
                for b00, b01, b10, b11 in [ (b00, b01, b10, b11) \
                                             for b01 in iter_b_01_10 \
                                             for b10 in iter_b_01_10 \
                                             for b00 in iter_b_00_11 \
                                             for b11 in iter_b_00_11 ]:

                    new_b = np.array([[cell_b[0][0]*b00 + cell_b[1][0]*b01,
                                       cell_b[0][1]*b00 + cell_b[1][1]*b01],
                                      [cell_b[0][0]*b10 + cell_b[1][0]*b11,
                                       cell_b[0][1]*b10 + cell_b[1][1]*b11]])

                    if new_b[0][0] > epsi and new_b[1][1] > epsi:
                        angle_b = pi/2 - atan(new_b[0][1]/new_b[0][0]) \
                                       - atan(new_b[1][0]/new_b[1][1])

                        if abs(angle_a - angle_b) < toll_rotation:
                            la0 = np.linalg.norm(new_a[0])
                            la1 = np.linalg.norm(new_a[1])
                            lb0 = np.linalg.norm(new_b[0])
                            lb1 = np.linalg.norm(new_b[1])

                            for diff0, diff1, invert in \
                              ((abs(la0-lb0)/la0, abs(la1-lb1)/la1, False),
                               (abs(la0-lb1)/la0, abs(la1-lb0)/la1, True)):

                                if diff0 < stretch_max and diff1 < stretch_max:

                                    deform_nrg = diff0**2 + diff1**2
                                    if len([ 1 for i in range(len(areas)) if \
                                       area < areas[i] - 1e-2 or \
                                       deform_nrg < deform_nrgs[i] - 1e-6]) \
                                       == len(areas):
                                        vect_a = [[a00, a01], [a10, a11]]
                                        vect_b = [[b00, b01], [b10, b11]]
                                        print('STRUCTURE', len(areas)+1)
                                        print('surface vectors a =', vect_a)
                                        print('surface vectors b =', vect_b)
                                        print('invert axis b  =', invert)
                                        print("deformation x' = {0:6.3f} %"\
                                            .format(diff0 * 100))
                                        print("deformation y' = {0:6.3f} %"\
                                            .format(diff1 * 100))
                                        print('slab area = {0:7.4f}\n'\
                                            .format(area))
                                        areas.append(area)
                                        deform_nrgs.append(deform_nrg)

                                        if deform_nrg == min(deform_nrgs):
                                            vect_a_opt = vect_a
                                            vect_b_opt = vect_b
                                            invert_opt = invert
                                            match_dimensions = True

    return vect_a_opt, vect_b_opt, invert_opt, match_dimensions

################################################################################
# CONVERT SURFACE ENERGY DICT
################################################################################

def convert_surface_energy_dict(surface_energy_dict):

    miller_list = []
    e_surf_list = []
    
    for miller_index in surface_energy_dict:
        e_surf_list.append(surface_energy_dict[miller_index])
        miller_list.append(convert_miller_index(miller_index))

    return miller_list, e_surf_list

################################################################################
# GET MOMENT OF INERTIA XYZ
################################################################################

def get_moments_of_inertia_xyz(atoms, center = None):

    if center is None:
        center = atoms.get_center_of_mass()

    positions = atoms.get_positions()-center
    masses = atoms.get_masses()

    I = np.zeros(3)

    for i in range(len(atoms)):

        x, y, z = positions[i]
        m = masses[i]

        I[0] += m * (y**2 + z**2)
        I[1] += m * (x**2 + z**2)
        I[2] += m * (x**2 + y**2)

    return I

################################################################################
# REORDER NEB IMAGES
################################################################################

def reorder_neb_images(first, last):

    coupled = cp.deepcopy(last)
    spared = [ a for a in last ]
    
    del coupled [ range(len(coupled)) ]
    
    for a in first:
        distances = [10.]*len(last)
        for b in [ b for b in last if b.symbol == a.symbol ]:
            distances[b.index] = np.linalg.norm(b.position-a.position)
        if np.min(distances) > 0.5:
            first += first.pop(i = a.index)
    
    for a in first:
        distances = [10.]*len(last)
        for b in [ b for b in last if b.symbol == a.symbol ]:
            distances[b.index] = np.linalg.norm(b.position-a.position)
        if np.min(distances) < 0.5:
            index = np.argmin(distances)
            coupled += last[index]
            spared[index] = None
    
    spared = Atoms([a for a in spared if a is not None])
    last = coupled + spared

    return first, last

################################################################################
# END
################################################################################
