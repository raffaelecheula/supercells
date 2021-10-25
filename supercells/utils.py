################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, cheula.raffaele@gmail.com
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
from ase.parallel import world, parprint, paropen
from qe_utils import reorder_neb_images
from ase_utils import (get_atom_list    ,
                       get_symbols_dict ,
                       array_from_dict  ,
                       atoms_fixed      ,
                       atoms_not_fixed  )
from supercell_builder import (cut_surface             ,
                               rotate_slab             ,
                               convert_miller_index    ,
                               check_inversion_symmetry)

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
        area_cell_a = cell_a[0][0]*cell_a[1][1]-cell_a[0][1]*cell_a[1][0]
        area_cell_b = cell_b[0][0]*cell_b[1][1]-cell_b[0][1]*cell_b[1][0]
        nmax_a = int(ceil(area_max/area_cell_a))
        nmax_b = int(ceil(area_max/area_cell_b))
    else:
        nmax_a = nmax_b = nmax

    vect_a_opt, vect_b_opt, invert_opt, match_dimensions = \
            match_slabs_dimensions_python(cell_a,
                                          cell_b,
                                          nmax_a,
                                          nmax_b,
                                          stretch_max,
                                          area_max,
                                          toll_rotation,
                                          epsi)

    if match_dimensions is False:
        print('NO MATCH BETWEEN THE SLABS IS FOUND!')
        return slab_a, slab_b

    slab_a = cut_surface(slab_a, surface_vectors = vect_a_opt)
    slab_a = rotate_slab(slab_a, 'automatic')
    slab_b = cut_surface(slab_b, surface_vectors = vect_b_opt)
    slab_b = rotate_slab(slab_b, 'automatic')
    
    if invert_opt is True:
        slab_b = rotate_slab(slab_b, 'invert axis')

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

    for a00, a01, a10, a11 in [ (a00, a01, a10, a11)
                                for a01 in iter_a_01_10
                                for a10 in iter_a_01_10
                                for a00 in iter_a_00_11
                                for a11 in iter_a_00_11 ]:

        new_a = np.array([[cell_a[0][0]*a00 + cell_a[1][0]*a01,
                           cell_a[0][1]*a00 + cell_a[1][1]*a01],
                          [cell_a[0][0]*a10 + cell_a[1][0]*a11,
                           cell_a[0][1]*a10 + cell_a[1][1]*a11]])

        area = new_a[0][0]*new_a[1][1] - new_a[0][1]*new_a[1][0]

        if new_a[0][0] > epsi and new_a[1][1] > epsi and area < area_max:

            angle_a = (pi/2 - atan(new_a[0][1]/new_a[0][0])
                            - atan(new_a[1][0]/new_a[1][1]))

            if minimum_angle < angle_a < pi - minimum_angle:
                for b00, b01, b10, b11 in [ (b00, b01, b10, b11)
                                             for b01 in iter_b_01_10
                                             for b10 in iter_b_01_10
                                             for b00 in iter_b_00_11
                                             for b11 in iter_b_00_11 ]:

                    new_b = np.array([[cell_b[0][0]*b00 + cell_b[1][0]*b01,
                                       cell_b[0][1]*b00 + cell_b[1][1]*b01],
                                      [cell_b[0][0]*b10 + cell_b[1][0]*b11,
                                       cell_b[0][1]*b10 + cell_b[1][1]*b11]])

                    if new_b[0][0] > epsi and new_b[1][1] > epsi:
                        angle_b = (pi/2 - atan(new_b[0][1]/new_b[0][0])
                                        - atan(new_b[1][0]/new_b[1][1]))

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
                                    if len([ 1 for i in range(len(areas))
                                             if area < areas[i] - 1e-2
                                             or deform_nrg < deform_nrgs[i]
                                             -1e-6]) == len(areas):
                                        vect_a = [[a00, a01], [a10, a11]]
                                        vect_b = [[b00, b01], [b10, b11]]
                                        print('STRUCTURE', len(areas)+1)
                                        print('surface vectors a =', vect_a)
                                        print('surface vectors b =', vect_b)
                                        print('invert axis b  =', invert)
                                        print("deformation x' = {0:6.3f} %"
                                            .format(diff0 * 100))
                                        print("deformation y' = {0:6.3f} %"
                                            .format(diff1 * 100))
                                        print('slab area = {0:7.4f}\n'
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
# WIRE CONSTRUCTION
################################################################################

def wire_construction(bulk            ,
                      diameter        ,
                      miller_indices  ,
                      surface_energies,
                      contact_index   = None,
                      adhesion_energy = None,
                      vacuum          = None,
                      size            = 25  ,
                      epsi            = 1e-4):

    for miller_index in miller_indices:
        miller_index = convert_miller_index(miller_index)

    if len(miller_indices) == 1 and miller_indices[0] == [0, 0, 1]:
        vector = [1, 0, 0]
    elif len(miller_indices) == 1:
        vector = [0, 0, 1]
    else:
        vector = miller_indices[1]
        assert(miller_indices[0] != miller_indices[1])

    direction = np.cross(miller_indices[0], vector)
    direction = np.abs(direction)

    surface_energies = np.array(surface_energies)/min(surface_energies)

    atoms = bulk[:]
    atoms *= size
    atoms.translate(-sum(atoms.cell)/2.)

    index_matrix = [[[+1, 0, 0], [0, +1, 0], [0, 0, +1]],
                    [[+1, 0, 0], [0, 0, +1], [0, +1, 0]],
                    [[0, +1, 0], [+1, 0, 0], [0, 0, +1]],
                    [[0, +1, 0], [0, 0, +1], [+1, 0, 0]],
                    [[0, 0, +1], [+1, 0, 0], [0, +1, 0]],
                    [[0, 0, +1], [0, +1, 0], [+1, 0, 0]],
                    [[-1, 0, 0], [0, +1, 0], [0, 0, +1]],
                    [[-1, 0, 0], [0, 0, +1], [0, +1, 0]],
                    [[0, -1, 0], [+1, 0, 0], [0, 0, +1]],
                    [[0, -1, 0], [0, 0, +1], [+1, 0, 0]],
                    [[0, 0, -1], [+1, 0, 0], [0, +1, 0]],
                    [[0, 0, -1], [0, +1, 0], [+1, 0, 0]],
                    [[+1, 0, 0], [0, -1, 0], [0, 0, +1]],
                    [[+1, 0, 0], [0, 0, -1], [0, +1, 0]],
                    [[0, +1, 0], [-1, 0, 0], [0, 0, +1]],
                    [[0, +1, 0], [0, 0, -1], [+1, 0, 0]],
                    [[0, 0, +1], [-1, 0, 0], [0, +1, 0]],
                    [[0, 0, +1], [0, -1, 0], [+1, 0, 0]],
                    [[+1, 0, 0], [0, +1, 0], [0, 0, -1]],
                    [[+1, 0, 0], [0, 0, +1], [0, -1, 0]],
                    [[0, +1, 0], [+1, 0, 0], [0, 0, -1]],
                    [[0, +1, 0], [0, 0, +1], [-1, 0, 0]],
                    [[0, 0, +1], [+1, 0, 0], [0, -1, 0]],
                    [[0, 0, +1], [0, +1, 0], [-1, 0, 0]],
                    [[-1, 0, 0], [0, -1, 0], [0, 0, +1]],
                    [[-1, 0, 0], [0, 0, -1], [0, +1, 0]],
                    [[0, -1, 0], [-1, 0, 0], [0, 0, +1]],
                    [[0, -1, 0], [0, 0, -1], [+1, 0, 0]],
                    [[0, 0, -1], [-1, 0, 0], [0, +1, 0]],
                    [[0, 0, -1], [0, -1, 0], [+1, 0, 0]],
                    [[-1, 0, 0], [0, +1, 0], [0, 0, -1]],
                    [[-1, 0, 0], [0, 0, +1], [0, -1, 0]],
                    [[0, -1, 0], [+1, 0, 0], [0, 0, -1]],
                    [[0, -1, 0], [0, 0, +1], [-1, 0, 0]],
                    [[0, 0, -1], [+1, 0, 0], [0, -1, 0]],
                    [[0, 0, -1], [0, +1, 0], [-1, 0, 0]],
                    [[+1, 0, 0], [0, -1, 0], [0, 0, -1]],
                    [[+1, 0, 0], [0, 0, -1], [0, -1, 0]],
                    [[0, +1, 0], [-1, 0, 0], [0, 0, -1]],
                    [[0, +1, 0], [0, 0, -1], [-1, 0, 0]],
                    [[0, 0, +1], [-1, 0, 0], [0, -1, 0]],
                    [[0, 0, +1], [0, -1, 0], [-1, 0, 0]],
                    [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                    [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
                    [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                    [[0, -1, 0], [0, 0, -1], [-1, 0, 0]],
                    [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],
                    [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]]

    for i in range(len(miller_indices)):
        index_vect = miller_indices[i]/np.linalg.norm(miller_indices[i])
        for j in range(len(index_matrix)):
            index_vector = np.dot(index_matrix[j], index_vect)
            if np.dot(index_vector, direction) < 1e-5:
                del atoms [[ a.index for a in atoms
                             if np.dot(a.position, index_vector) > \
                             diameter/2*surface_energies[i] ]]

    if contact_index is not None:
        diagonal = np.linalg.norm(contact_index[:2])
        if diagonal:
            rotation_angle = atan(contact_index[2]/diagonal)
            atoms.rotate(-rotation_angle*180/pi, direction)
        else:
            atoms.rotate(90, direction)

    if np.allclose(direction, [0, 0, 1]):
        atoms.rotate(-90, 'y')
    else:
        y_axis = np.cross([0, 0, 1], direction)
        z_axis = np.cross(direction, y_axis)
        atoms.set_cell([direction, y_axis, z_axis], scale_atoms = False)
        atoms.center(vacuum = 0.)
        x_axis = np.array([1, 0, 0])*np.linalg.norm(atoms.cell[0])
        y_axis = np.array([0, 1, 0])*np.linalg.norm(atoms.cell[1])
        z_axis = np.array([0, 0, 1])*np.linalg.norm(atoms.cell[2])
        atoms.set_cell([x_axis, y_axis, z_axis], scale_atoms = True)

    atoms.center(vacuum = 0.)

    del atoms[[a.index for a in atoms if a.position[0] < atoms.cell[0][0]*1/4 \
        or a.position[0] > atoms.cell[0][0]*3/4]]

    atoms.center(vacuum = 0.)
    args = np.argsort(atoms.positions[:, 0])
    atoms = atoms[args]
    
    position_zero = atoms[0].position

    for a in atoms[1:]:
        if np.linalg.norm(a.position[1:]-position_zero[1:]) < epsi :
            check = 0
            length = a.position[0]-position_zero[0]
            for b in [ b for b in atoms if b.position[0] < length ]:
                for c in [ c for c in atoms if c.position[0] >= length-epsi ]:
                    if (c.position[0]-b.position[0]-length < epsi and
                        np.linalg.norm(c.position[1:]-b.position[1:])) < epsi:
                        check += 1
                        break

            if len([ b for b in atoms if b.position[0] < length ]) == check:
                break

    del atoms [[ a.index for a in atoms if a.position[0] > length-epsi ]]

    atoms.cell[0][0] = length

    if adhesion_energy is not None:
        height = atoms.cell[2][2]/2.-diameter/2*adhesion_energy
        del atoms [[ a.index for a in atoms if a.position[2] < height ]]

    if vacuum is not None:
        atoms.center(vacuum = vacuum/2., axis = (1, 2))

    return atoms

################################################################################
# CREATE INTERFACE WIRE
################################################################################

#def create_interface_wire(support, wire, distance = 1.5, vacuum = 10.,
#                          x_min = 10., y_min = 10., stretch_max = 0.1,
#                          nmax_a = 5, epsi = 1e-4):
#
#    atoms = support[:]
#    wire = wire[:]
#
#    atoms = rotate_slab(atoms, 'automatic')
#
#    cell_a = np.vstack([atoms.cell[:2][0][:2], atoms.cell[:2][1][:2]])
#
#    angle = pi/2 - atan(cell_a[0][1]/cell_a[0][0]) \
#                 - atan(cell_a[1][0]/cell_a[1][1])
#
#    if not abs(angle - pi/2) < epsi:
#
#        match_angle = False
#        
#        new_a = np.copy(cell_a)
#
#        iter_a_11 = [ i for i in range(1, nmax_a+1) ]
#
#        index_a = [0]
#        for i in range(1, nmax_a): index_a += [i, -i]
#        iter_a_10 = [ i for i in index_a ]
#        
#        for a_10, a_11 in [ (a_10, a_11) for a_10 in iter_a_10 \
#                                         for a_11 in iter_a_11 ]:
#
#            new_a[1] = [cell_a[0][0]*a_10 + cell_a[1][0]*a_11,
#                        cell_a[0][1]*a_10 + cell_a[1][1]*a_11]
#
#            angle = pi/2 - atan(new_a[0][1]/new_a[0][0]) \
#                         - atan(new_a[1][0]/new_a[1][1])
#
#            if new_a[1][1] > epsi and abs(angle - pi/2) < epsi:
#                match_angle = True
#                break
#        
#        if match_angle is False:
#            print('NO RECTANGULAR SLAB FOUND!')
#        else:
#            atoms = cut_surface(atoms, surface_vectors = [[1, 0], [a_10, a_11]])
#
#    stretch = abs(atoms.cell[0][0] - wire.cell[0][0])
#
#    i = 0
#    while stretch > stretch_max:
#
#        n_a_x = int(np.round(x_min / atoms.cell[0][0])) + i
#        n_a_y = int(np.round(y_min / atoms.cell[0][0]))
#
#        n_b = int(np.round(atoms.cell[0][0]*n_a_x / wire.cell[0][0]))
#    
#        stretch = abs(atoms.cell[0][0]*n_a_x - wire.cell[0][0]*n_b)
#
#        print('repetitions of wire cell in x direction: {}'.format(n_a_x))
#        print('stretching of the slab: {0:7.4f} A\n'.format(stretch))
#
#        i += 1
#    
#    atoms *= (n_a_x, n_a_y, 1)
#    wire *= (n_b, 1, 1)
#
#    atoms.center(vacuum = 0., axis = 2)
#    wire.center(vacuum = 0., axis = 2)
#
#    height = atoms.cell[2][2]
#
#    wire.set_cell(np.vstack([atoms.cell[0], wire.cell[1:]]), scale_atoms = True)
#    wire.translate([0., (atoms.cell[1][1] - wire.cell[1][1]) / 2,
#                    height + distance])
#
#    atoms += wire
#    atoms.center(vacuum = vacuum / 2., axis = 2)
#
#    return atoms

def create_interface_wire(support,
                          wire,
                          distance      = 1.5,
                          vacuum        = 10.,
                          x_min         = 10.,
                          y_min         = 10.,
                          stretch_max   = 0.1,
                          nmax_a        = 5,
                          nmax_b        = 5,
                          minimum_angle = pi/5,
                          epsi          = 1e-4):

    atoms = support[:]
    wire = wire[:]

    atoms = rotate_slab(atoms, 'automatic')

    cell_a = np.vstack([atoms.cell[:2][0][:2], atoms.cell[:2][1][:2]])

    iter_a_00_11 = [ i for i in range(1, nmax_a+1) ]

    index_a = [0]
    for i in range(1, nmax_a): index_a += [i, -i]
    iter_a_01_10 = [ i for i in index_a ]

    iter_b_00 = [ i for i in range(1, nmax_b+1) ]

    areas = [1e10]
    stretchings = [1e10]
    #area_opt = 1e10
    #stretch_opt = 1e10

    j = 0
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

        if new_a[0][0] > x_min and new_a[1][1] > y_min:

            angle_a = pi/2 - atan(new_a[0][1]/new_a[0][0]) \
                           - atan(new_a[1][0]/new_a[1][1])

            if minimum_angle < angle_a < pi - minimum_angle:
                
                #i = 0
                #for invert in (False, True):
                #
                #    b_00 = int(round(np.linalg.norm(new_a[i])/wire.cell[1][1]))
                #    stretch = abs(np.linalg.norm(new_a[i])-b_00*wire.cell[1][1])
                #    i = 1

                b_00 = int(round(np.linalg.norm(new_a[0])/wire.cell[0][0]))
                stretch = abs(np.linalg.norm(new_a[0])-b_00*wire.cell[0][0])

                if stretch < stretch_max:
                
                    vect_a = [[a00, a01], [a10, a11]]
                    print('STRUCTURE', j+1)
                    print('surface vectors support =', vect_a)
                    print('repetitions wire =', b_00)
                    #print('invert axis b =', invert)
                    print('wire stretching = {0:6.3f} A'.format(stretch))
                    print('slab area = {0:7.4f}\n'.format(area))
                    
                    areas.append(area)
                    
                    if area == min(areas):
                        stretchings.append(stretch)
                        if stretch == min(stretchings):
                            vect_a_opt = vect_a
                            b_00_opt = b_00
                            #invert_opt = invert
                    
                    #if area > area_opt:
                    #    break
                    #if stretch >= stretch_opt:
                    #    break
                    #else:
                    #    area_opt = area
                    #    stretch_opt = stretch
                    #    vect_a_opt = vect_a
                    #    b_00_opt = b_00
                    #    #invert_opt = invert

    atoms = cut_surface(atoms, surface_vectors = vect_a_opt)
    atoms = rotate_slab(atoms, 'automatic')

    wire *= (b_00_opt, 1, 1)

    atoms.center(vacuum = 0., axis = 2)
    wire.center(vacuum = 0., axis = 2)

    height = atoms.cell[2][2]

    print('FINAL STRUCTURE')
    print('surface vectors a =', vect_a_opt)
    print('repetitions wire =', b_00_opt)
    #print('invert axis b =', invert_opt, '\n')

    stretch_opt = atoms.cell[0][0]-wire.cell[0][0]
    print('wire stretching = {0:6.3f} A'.format(stretch_opt))
    print('area = {0:6.3f} A'.format(atoms.cell[0][0]*atoms.cell[1][1]))

    wire.set_cell(np.vstack([atoms.cell[0], wire.cell[1:]]), scale_atoms = True)
    wire.translate([0., (atoms.cell[1][1] - wire.cell[1][1]) / 2,
                    height + distance])

    atoms += wire
    atoms = cut_surface(atoms)
    atoms.center(vacuum = vacuum / 2., axis = 2)

    return atoms

################################################################################
# END
################################################################################
