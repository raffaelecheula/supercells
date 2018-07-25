################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

from __future__ import absolute_import, division, print_function
import ast, re
import numpy as np
from math import sin, pi, sqrt, atan, ceil
from functools import reduce
from fractions import gcd
from collections import OrderedDict
from ase import Atom, Atoms
from ase.io import read
from ase.units import kB, create_units
from ase.constraints import FixAtoms, FixCartesian
from ase.geometry import get_duplicate_atoms
from supercell_builder import (cut_surface, rotate_slab, convert_miller_index,
                               check_inversion_symmetry)

################################################################################
# GET ATOM LIST
################################################################################

def get_atom_list(atoms):

    symbols = atoms.get_chemical_symbols()
    if len(symbols) > 1:
        for i in range(len(symbols)-1, 0, -1):
            for j in range(i):
                if symbols[j] == symbols[i]:
                    del symbols[i]
                    break

    return symbol

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

def read_vib_energies(filename = 'log.txt'):

    vib_energies = []
    f = open(filename, 'rU')
    
    lines = f.readlines()
    for i in range(3, len(lines) - 2):
        try: vib_energies.append(float(lines[i].split()[1]) * 1e-3)
        except: pass

    return vib_energies

################################################################################
# WRITE NEB DAT
################################################################################

def write_neb_dat(input_data_neb, filename = 'neb.dat', mode = 'w+'):

    neb_dict = OrderedDict([['string_method'       , None],
                            ['restart_mode'        , None],
                            ['nstep_path'          , None],
                            ['num_of_images'       , None],
                            ['opt_scheme'          , None],
                            ['CI_scheme'           , None],
                            ['first_last_opt'      , None],
                            ['minimum_image'       , None],
                            ['temp_req'            , None],
                            ['ds'                  , None],
                            ['k_max'               , None],
                            ['k_min'               , None],
                            ['path_thr'            , None],
                            ['use_masses'          , None],
                            ['use_freezing'        , None],
                            ['lfcpopt'             , None],
                            ['fcp_mu'              , None],
                            ['fcp_tot_charge_first', None],
                            ['fcp_tot_charge_last' , None]])

    for arg in input_data_neb:
        neb_dict[arg] = input_data_neb[arg]

    f = open(filename, mode)

    f.write('&PATH\n')
    for arg in [ arg for arg in neb_dict if neb_dict[arg] is not None ]:
        if isinstance(neb_dict[arg], str):
            neb_dict[arg] = '\'' + neb_dict[arg]+ '\''
        elif neb_dict[arg] is True:
            neb_dict[arg] = '.true.'
        elif neb_dict[arg] is False:
            neb_dict[arg] = '.false.'
        f.write('   {0} = {1}\n'.format(str(arg).ljust(16), neb_dict[arg]))
    f.write('/')
    f.close()

################################################################################
# WRITE NEB INP
################################################################################

def write_neb_inp(input_data_neb, images, calc, filename = 'neb.inp'):

    f = open(filename, 'w+')
    
    f.write('BEGIN\n')
    f.write('BEGIN_PATH_INPUT\n')
    f.close()

    write_neb_dat(input_data_neb, filename, mode = 'a+')

    f = open(filename, 'a+')
    f.write('\nEND_PATH_INPUT\n')
    f.write('BEGIN_ENGINE_INPUT\n')

    for i in range(len(images)):

        calc.write_input(images[i])

        g = open('espresso.pwi', 'rU')
        lines = g.readlines()
        g.close()
        
        for n, line in enumerate(lines):
            if 'ATOMIC_POSITIONS' in line:
                atomic_positions_line = n
                break

        if i == 0:
            for line in lines[:n]:
                f.write(line)
            f.write('BEGIN_POSITIONS\n')
            f.write('FIRST_IMAGE\n')
        elif i == len(images)-1:
            f.write('LAST_IMAGE\n')
        else:
            f.write('INTERMEDIATE_IMAGE\n')

        for line in lines[n:]:
            f.write(line)

    f.write('END_POSITIONS\n')
    f.write('END_ENGINE_INPUT\n')
    f.write('END\n')

    f.close()

################################################################################
# PRINT AXSF
################################################################################

def print_axsf(filename, animation):

    f = open(filename, 'w+')

    cell = animation[0].cell
    possible_types = get_atom_list(animation[0])

    print(' ANIMSTEP', len(animation), file = f)
    print(' CRYSTAL', file = f)
    print(' PRIMVEC', file = f)
    print("{0:14.8f} {1:14.8f} {2:14.8f}".format(cell[0][0], cell[0][1],
          cell[0][2]), file = f)
    print("{0:14.8f} {1:14.8f} {2:14.8f}".format(cell[1][0], cell[1][1],
          cell[1][2]), file = f)
    print("{0:14.8f} {1:14.8f} {2:14.8f}".format(cell[2][0], cell[2][1],
          cell[2][2]), file = f)

    for i, atoms in enumerate(animation):
        print(' PRIMCOORD', i + 1, file = f)
        print(len(atoms), len(possible_types), file = f)
        for a in atoms:
            print("{0:3s} {1:14.8f} {2:14.8f} {3:14.8f}".format(a.symbol, 
                  a.position[0], a.position[1], a.position[2]), file = f)
    f.close()

################################################################################
# READ AXSF
################################################################################

def read_axsf(filename):

    fileobj = open(filename, 'rU')
    lines = fileobj.readlines()

    for line in lines:
        if 'PRIMCOORD' in line:
            key = 'PRIMCOORD'
            break
        elif 'ATOMS' in line:
            key = 'ATOMS'
            break

    if key == 'PRIMCOORD':
        for n, line in enumerate(lines):
            if 'PRIMVEC' in line:
                break
        cell_vectors = np.zeros((3, 3))
        for i, line in enumerate(lines[n + 1 : n + 4]):
            entries = line.split()
            cell_vectors[i][0] = float(entries[0])
            cell_vectors[i][1] = float(entries[1])
            cell_vectors[i][2] = float(entries[2])
        atoms_zero = Atoms(cell = cell_vectors, pbc = (True, True, True))
        increment = 2

    elif key == 'ATOMS':
        atoms_zero = Atoms(pbc = (False, False, False))
        increment = 1

    key = 'PRIMCOORD'
    animation = []
    for n, line in enumerate(lines):
        if key in line:
            atoms = Atoms(cell = cell_vectors, pbc = (True, True, True))
            for line in lines[ n + increment : ]:
                entr = line.split()
                if entr[0] == key:
                    break
                symbol = entr[0]
                position = (float(entr[1]), float(entr[2]), float(entr[3]))
                atoms += Atom(symbol, position = position)
                animation += [ atoms ]
    f.close()

    return animation

################################################################################
# READ MODES AXSF
################################################################################

def write_modes_axsf(vib, kT = kB * 300, nimages = 30):

    for index, energy in enumerate(vib.get_energies()):

        if abs(energy) > 1e-5:
        
            animation = []

            mode = vib.get_mode(index) * sqrt(kT / abs(vib.hnu[index]))
            p = vib.atoms.positions.copy()
            index %= 3 * len(vib.indices)
            for x in np.linspace(0, 2 * pi, nimages, endpoint = False):
                vib.atoms.set_positions(p + sin(x) * mode)
                animation += [vib.atoms.copy()]
            vib.atoms.set_positions(p)

            print_axsf('mode_{}.axsf'.format(str(index)), animation)

################################################################################
# READ NEB CRD
################################################################################

def read_neb_crd(images, filename = 'pwscf.crd'):

    fileobj = open(filename, 'rU')
    lines = fileobj.readlines()

    n_atoms = len(images[0])
    n_images = len(images)

    num = 2
    for i, image in enumerate(images):
        positions = []
        for line in lines[num:num+n_atoms]:
            positions.append(line.split()[1:4])
        image.set_positions(positions)
        num += n_atoms+2

    return images

################################################################################
# READ QUANTUM ESPRESSO OUT
################################################################################

def read_qe_out(filename):

    units = create_units('2006')

    atoms = Atoms(pbc = True)
    cell = np.zeros((3, 3))

    fileobj = open(filename, 'rU')
    lines = fileobj.readlines()

    atomic_positions = 'angstrom'
    for n, line in enumerate(lines):
        if 'ATOMIC_POSITIONS (crystal)' in line:
            atomic_positions = 'crystal'
        elif 'celldm(1)' in line:
            celldm = float(line.split()[1]) * units['Bohr']
        elif 'crystal axes: (cart. coord. in units of alat)' in line:
            n_cell = n
        elif '!' in line:
            n_nrg = n
        elif 'Final energy' in line:
            n_fin = n
        elif 'Begin final coordinates' in line:
            n_pos = n

    for i in range(3):
        line = lines[n_cell+1+i]
        cell[i] = [ float(c)*celldm for c in line.split()[3:6] ]

    if str(lines[n_pos+3].split()[0]) == 'CELL_PARAMETERS':
        for i in range(3):
            line = lines[n_pos+4+i]
            cell[i] = [ float(c)*celldm for c in line.split()[3:6] ]
        n_pos += 6

    atoms.set_cell(cell)

    energy = float(lines[n_fin].split()[3]) * units['Ry']

    index = 0
    indices = []
    constraints = []
    translate_constraints = {0: True, 1: False}
    for line in lines[n_pos+3:]:
        if line.split()[0] == 'End':
            break
        symbol = line.split()[0]
        positions = [[ float(i) for i in line.split()[1:4] ]]
        fix = [ translate_constraints[int(i)] for i in line.split()[4:] ]

        if atomic_positions is 'crystal':
            atoms += Atoms(symbol, scaled_positions = positions)
        else:
            atoms += Atoms(symbol, positions = positions)

        if fix == [True, True, True]:
            indices.append(index)
        elif fix != []:
            constraints.append(FixCartesian([index], fix))
        index += 1

    atoms_ase = read(filename)
    atoms_ase.set_chemical_symbols(atoms.get_chemical_symbols())
    atoms_ase.set_positions(atoms.get_positions())

    constraints.append(FixAtoms(indices = indices))
    atoms_ase.set_constraint(constraints)
    
    return atoms_ase

################################################################################
# READ QUANTUM ESPRESSO INP
################################################################################

def read_qe_inp(filename):

    fileobj = open(filename, 'rU')
    lines = fileobj.readlines()

    n_as = 0
    n_kp = 0
    gamma = False
    for n, line in enumerate(lines):
        if 'ATOMIC_SPECIES' in line:
            n_as = n
        elif 'K_POINTS' in line:
            if 'gamma' in line:
                gamma = True
            n_kp = n

    input_data = {}
    for n, line in enumerate(lines):
        if 'ATOMIC_SPECIES' in line or 'ATOMIC_POSITIONS' in line or \
           'K_POINTS' in line or 'CELL_PARAMETERS' in line:
            break
        if len(line.strip()) == 0 or line is '\n':
            pass
        elif line[0] in ('&', '/'):
            pass
        else:
            keyword, argument = line.split('=')
            keyword = re.sub(re.compile(r'\s+'), '', keyword)
            argument = re.sub(re.compile(r'\s+'), '', argument)
            if '.true.' in argument:
                argument = True
            elif '.false.' in argument:
                argumnt = False
            else:
                argument = ast.literal_eval(argument)
            if type(argument) is tuple: argument = argument[0]
            input_data[keyword] = argument

    pseudos = {}
    for n, line in enumerate(lines[n_as+1:]):
        if len(line.strip()) == 0 or line is '\n':
            break
        element, MW, pseudo = line.split()
        pseudos[element] = pseudo

    if gamma:
        kpts = (1, 1, 1)
    else:
        kpts = [ int(i) for i in lines[n_kp+1].split()[:3] ]
        koffset = [ int(i) for i in lines[n_kp+1].split()[3:] ]

    try: del input_data['calculation']
    except: pass
    try: del input_data['restart_mode']
    except: pass
    try: del input_data['max_seconds']
    except: pass

    return input_data, pseudos, kpts, koffset

################################################################################
# UPDATE PSEUDOPOTENTIALS
################################################################################

def update_pseudos(pseudos, filename):

    input_data, pseudos_new, kpts, koffset = read_qe_inp(filename)
    
    return dict(pseudos.items() + pseudos_new.items())

################################################################################
# ATOMS FIXED
################################################################################

def atoms_fixed(atoms):

    fixed = np.concatenate([ a.__dict__['index'] for a in atoms.constraints if \
                             a.__class__.__name__ == 'FixAtoms' ])

    return fixed

################################################################################
# ATOMS NOT FIXED
################################################################################

def atoms_not_fixed(atoms):

    fixed = atoms_fixed(atoms)
    not_fixed = [ i for i in range(len(atoms)) if i not in fixed ]
    
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

    slab_a = cut_surface(slab_a, vect_a_opt[0], vect_a_opt[1])
    slab_a = rotate_slab(slab_a, 'automatic')
    slab_b = cut_surface(slab_b, vect_b_opt[0], vect_b_opt[1])
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
# END
################################################################################
