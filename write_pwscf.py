#!/usr/bin/python

from __future__ import division, print_function
import ase
import numpy as np
from ase import Atom, Atoms
from ase.constraints import FixAtoms, FixScaled

################################################################################
#                                 PWSCF CLASS
################################################################################

class PWscf:
    def __init__(self):
        self.atoms                = Atoms('X')
        self.control              = Control()
        self.system               = System(self.atoms)
        self.magnet               = StartingMagnetization(self.atoms)
        self.lda_plus_u           = LdaPlusU(self.atoms)
        self.electrons            = Electrons()
        self.ions                 = Ions()
        self.cell                 = Cell()
        self.atomic_species       = AtomicSpecies(self.atoms)
        self.atomic_positions     = AtomicPositions()
        self.kpoints              = Kpoints()

    def write_input(self, filename):

        system_vars = [ (key, var) for key, var in \
            vars(self.system).items() if key not in ('nat', 'ntyp') ]
        magnet_vars = [ (key, var) for key, var in \
            vars(self.magnet).items() if key is not 'symbol' ]
        lda_plus_u_vars = [ (key, var) for key, var in \
            vars(self.lda_plus_u).items() if key is not 'symbol' ]
        atomic_species_vars = [ (key, var) for key, var in \
            vars(self.atomic_species).items() if key not in ('symbol', 'ntyp') \
            if str(var) != str('[ 1.]') if str(var) != str([b'X.UPF']) ]

        exec('self.system = System(self.atoms)')
        exec('self.magnet = StartingMagnetization(self.atoms)')
        exec('self.lda_plus_u = LdaPlusU(self.atoms)')
        exec('self.atomic_species = AtomicSpecies(self.atoms)')

        for key, val in system_vars:
            exec('self.system.' + key + ' = val')
        for key, val in magnet_vars:
            exec('self.magnet.' + key + ' = val')
        for key, val in lda_plus_u_vars:
            exec('self.lda_plus_u.' + key + ' = val')
        for key, val in atomic_species_vars:
            exec('self.atomic_species.' + key + ' = val')
 
        write_pwscf_input(self, filename)

class Control:
    def __init__(self):
        self.calculation            = 'scf'
        self.restart_mode           = 'from_scratch'

class System:
    def __init__(self, atoms):
        self.ibrav                  = 0
        self.nat                    = len(atoms)
        symbol                      = get_atom_list(atoms)
        self.ntyp                   = len(symbol)
        self.nspin                  = None
        self.lda_plus_u             = None

class StartingMagnetization:
    def __init__(self, atoms):
        symbol                      = get_atom_list(atoms)
        self.symbol                 = symbol
        self.mag_dict               = None
        self.starting_magnetization = None

class LdaPlusU:
    def __init__(self, atoms):
        symbol                      = get_atom_list(atoms)
        self.symbol                 = symbol
        self.u_dict                 = None
        self.Hubbard_U              = None

class Electrons:
    def __init__(self):
        self.electron_maxstep       = None

class Ions:
    def __init__(self):
        self.ion_dynamics           = None

class Cell:
    def __init__(self):
        self.cell_dynamics          = None

class AtomicSpecies:
    def __init__(self, atoms):
        symbol                      = get_atom_list(atoms)
        mass, pseudo                = set_mass_and_pseudo(symbol)
        ntyp                        = len(symbol)
        self.ntyp                   = ntyp
        self.symbol                 = symbol
        self.mass                   = mass
        self.pseudo_potential       = pseudo

class AtomicPositions:
    def __init__(self):
        self.type                   = 'angstrom'

class Kpoints:
    def __init__(self):
        self.type                   = 'automatic'
        self.nk                     = [1, 1, 1]
        self.sk                     = [0, 0, 0]

################################################################################
#                            GET REDUCE ATOM LIST
################################################################################

def get_atom_list(atoms):
    symbol = atoms.get_chemical_symbols()
    if len (symbol) > 1:
        for i in range(len(symbol)-1, 0, -1):
            for j in range(i):
                if symbol[j] == symbol[i]:
                    del symbol[i]
                    break
    return symbol

################################################################################
#                             SET MASS AND PSEUDO
################################################################################

def set_mass_and_pseudo(symbol):
    mass = np.zeros(len(symbol), dtype = np.float)
    for i in range(len(symbol)):
        mass[i] = Atom(symbol[i]).mass
    pseudo = np.zeros(len(symbol), dtype='S30')
    for i in range(len(symbol)):
        pseudo[i] = symbol[i] + '.UPF'
    return mass, pseudo

################################################################################
#                               ARRAY FROM DICT
################################################################################

def array_from_dict(symbol, array_dict):
    array = [None]*len(symbol)
    for i in range(len(symbol)):
        if symbol[i] in array_dict:
            array[i] = array_dict[symbol[i]]
    return array

################################################################################
#                                 WRITE KEY
################################################################################

def write_key(item, dict):

    value = vars(dict) [item]
    if type(value) is str:
        str_value = '\'' + value + '\','
    if type(value) is float:
        str_value = str(value) + ','
    if type(value) is int:
        str_value = str(value) + ','
    if type(value) is bool:
        if (value):
            str_value = '.true.,'
        else:
            str_value = '.false.,'
    item_len = item.__len__()
    default_len = 25
    add_str = ''
    for i in range(item_len, default_len):
        add_str +=' ' 
    string = '  ' + item + add_str + ' = ' + str_value
    return string

################################################################################
#                               WRITE ARRAY KEY
################################################################################

def write_array_key(item, dict, f):

    array_value = vars(dict) [item]
    for i in range(len(array_value)):
        value = array_value[i]
        item_len = item.__len__()
        default_len = 25
        add_str = ''
        for j in range(item_len + 3, default_len):
            add_str += ' '
 
        string = '  ' + item + '('+str(i+1)+')' + add_str + ' = ' + \
                 str(value) + ','
        if value is not None:
            print(string, file = f)

################################################################################
#                             WRITE ATOMIC SPECIES
################################################################################

def write_atomic_species(atomic_species, f):

    for i in range(atomic_species.ntyp):
        print("%3s %9.4f %7s" % ( \
              atomic_species.symbol[i] , \
              atomic_species.mass[i], \
              atomic_species.pseudo_potential[i]), file = f)

################################################################################
#                              WRITE STRUCTURE
################################################################################

def write_structure(atoms, atomic_positions, f):

    print('ATOMIC_POSITIONS (' + atomic_positions.type + ')', file = f)
    sflags = np.zeros((len(atoms), 3), dtype = bool)
    newsflags = np.ones((len(atoms), 3), dtype = np.int)
    if atoms.constraints:
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                sflags [constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                sflags [constr.index] = [True, True, True]
            for i in range(len(atoms)):
                for j in range(3):
                    if  sflags [i,j]:
                        newsflags [i,j] = 0

    for i in range(len(atoms)):
        if atomic_positions.type is 'angstrom':
            x_pos = atoms.get_positions()[i,0]
            y_pos = atoms.get_positions()[i,1]
            z_pos = atoms.get_positions()[i,2]
        elif atomic_positions.type is 'crystal':
            x_pos = atoms.get_scaled_positions()[i,0]
            y_pos = atoms.get_scaled_positions()[i,1]
            z_pos = atoms.get_scaled_positions()[i,2]
        print('%3s %14.9f %14.9f %14.9f %5i %3i %3i' %( \
              atoms.get_chemical_symbols()[i], x_pos, y_pos, z_pos, \
              newsflags[i,0], newsflags[i,1], newsflags[i,2]), file = f)

    print('', file = f)
    print('CELL_PARAMETERS (angstrom)', file = f)
    for i in range(3):
        print('%14.9f %14.9f %14.9f' %( \
              atoms.cell[i,0], \
              atoms.cell[i,1], \
              atoms.cell[i,2]), file = f)

################################################################################
#                                WRITE K POINTS
################################################################################

def write_k_points(kpoints, f):
    if kpoints.type.lower() == 'gamma':
        kpoints.nk[:] = 1
        kpoints.sk[:] = 0
        
    print('K_POINTS (' + kpoints.type + ')', file = f)
    print("%3i %3i %3i %3i %3i %3i" % ( 
          kpoints.nk[0] ,  kpoints.nk[1] ,  kpoints.nk[2], 
          kpoints.sk[0] ,  kpoints.sk[1] ,  kpoints.sk[2]), file = f)

################################################################################
#                              WRITE PWSCF INPUT
################################################################################

def write_pwscf_input(object, filename):

    f = open(filename, 'w')
    
    """ &CONTROL section """
    print('&CONTROL', file = f)
    dict = object.control
    for item in sorted(vars(dict)):
        if vars(dict) [item] is not None:
            print(write_key(item, dict), file = f)

    print('/', file = f)

    """ &SYSTEM section """
    print('&SYSTEM', file = f)
    dict = object.system
    for item in sorted(vars(dict)):
        if vars(dict) [item] is not None:
            print(write_key(item, dict), file = f)

    if object.system.nspin is 2:
        dict = object.magnet
        if dict.mag_dict is not None:
            dict.starting_magnetization = \
                 array_from_dict(dict.symbol, dict.mag_dict)
        if dict.starting_magnetization is not None:
            write_array_key('starting_magnetization', dict, f)

    if object.system.lda_plus_u is True:
        dict = object.lda_plus_u
        if dict.u_dict is not None:
            dict.Hubbard_U = array_from_dict(dict.symbol, dict.u_dict)
        if dict.Hubbard_U is not None:
            write_array_key('Hubbard_U', dict, f)

    print('/', file = f)

    """ &ELECTRONS section """
    print('&ELECTRONS', file = f)
    dict = object.electrons
    for item in sorted(vars(dict)):
        if vars(dict) [item] is not None:
            print(write_key(item, dict), file = f)

    print('/', file = f)

    """ &IONS section """
    if object.control.calculation in ('relax', 'md', 'vc-relax', 'vc-md'):
        print('&IONS', file = f)
        dict = object.ions
        for item in sorted(vars(dict)):
            if vars(dict) [item] is not None:
                print(write_key(item, dict), file = f)

        print('/', file = f)

    """ &CELL section """
    if object.control.calculation in ('vc-relax', 'vc-md'):
        print('&CELL', file = f)
        dict = object.cell
        for item in sorted(vars(dict)):
            if vars(dict) [item] is not None:
                print(write_key(item, dict), file = f)

        print('/', file = f)

    """ ATOMIC_SPECIES section """
    print('ATOMIC_SPECIES', file = f)
    write_atomic_species(object.atomic_species, f)
    print('', file = f)
    write_structure(object.atoms, object.atomic_positions, f)
    print('', file = f)
    write_k_points(object.kpoints, f)

################################################################################
#                                    END
################################################################################
