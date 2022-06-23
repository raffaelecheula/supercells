################################################################################
# SUPERCELL BUILDER
# Distributed under the GPLv3 license
# Author: Raffaele Cheula
# cheula.raffaele@gmail.com
# Laboratory of Catalysis and Catalytic Processes (LCCP)
# Department of Energy, Politecnico di Milano
################################################################################

from __future__ import absolute_import, division, print_function
import numpy as np
import copy as cp
import ase.build
from math import pi, sqrt, ceil, atan
from ase import Atoms
from ase.io import read
from ase.constraints import FixAtoms

################################################################################
# BULK CLASS
################################################################################

class Bulk:
    
    def __init__(self,
                 bulk_type         = None,
                 input_bulk        = None,
                 elements          = None,
                 lattice_constants = None,
                 kpts_bulk         = None):

        if isinstance(elements, str):
            elements = [elements]

        if isinstance(lattice_constants, (int, float)):
            lattice_constants = [lattice_constants]

        if input_bulk is not None:
            atoms = import_bulk_structure(input_bulk)
        else:
            atoms = build_bulk_structure(bulk_type        ,
                                         elements         ,
                                         lattice_constants)

        koffset = (0, 0, 0)

        self.atoms             = atoms
        self.bulk_type         = bulk_type
        self.input_bulk        = input_bulk
        self.elements          = elements
        self.lattice_constants = lattice_constants
        self.kpts_bulk         = kpts_bulk
        self.koffset           = koffset

    # -------------------------------------------------------------------
    #  UPDATE
    # -------------------------------------------------------------------

    def update(self):

        bulk = Bulk(bulk_type         = self.bulk_type        ,
                    input_bulk        = self.input_bulk       ,
                    elements          = self.elements         ,
                    lattice_constants = self.lattice_constants,
                    kpts_bulk         = self.kpts_bulk        )

        return bulk

################################################################################
# SLAB CLASS
################################################################################

class Slab:

    def __init__(self,
                 bulk            = None  ,
                 input_slab      = None  ,
                 miller_index    = None  ,
                 surface_vectors = None  ,
                 dimensions      = (1, 1),
                 layers          = 1     ,
                 layers_fixed    = None  ,
                 symmetry        = None  ,
                 rotation_angle  = None  ,
                 cut_top         = None  ,
                 cut_bottom      = None  ,
                 adsorbates      = []    ,
                 vacancies       = []    ,
                 scale_kpts      = None  ,
                 vacuum          = None  ,
                 sort_atoms      = False ):

        layers = int(layers)
        
        if surface_vectors == 'automatic':
            if input_slab is not None:
                surface_vectors = None
            else:
                surface_vectors = automatic_vectors(bulk.bulk_type,
                                                    miller_index  )

        repetitions = (1, 1)
        if surface_vectors is not None:
            surface_vectors = [np.dot(surface_vectors[0], dimensions[0]),
                               np.dot(surface_vectors[1], dimensions[1])]
        else:
            repetitions = (int(dimensions[0]), int(dimensions[1]))

        if input_slab is not None:
            atoms = import_slab_structure(input_slab = input_slab ,
                                          dimensions = repetitions,
                                          vacuum     = vacuum     )
        else:
            atoms = build_slab_structure(atoms     = bulk.atoms            ,
                                 bulk_type         = bulk.bulk_type        ,
                                 elements          = bulk.elements         ,
                                 lattice_constants = bulk.lattice_constants,
                                 miller_index      = miller_index          ,
                                 dimensions        = repetitions           ,
                                 layers            = layers                )

        if surface_vectors is not None:
            atoms = cut_surface(atoms           = atoms          , 
                                surface_vectors = surface_vectors)

        if rotation_angle is not None:
            atoms = rotate_slab(atoms          = atoms         ,
                                rotation_angle = rotation_angle)

        if cut_top is not None:
            atoms = cut_top_slab(atoms   = atoms  ,
                                 cut_top = cut_top, 
                                 vacuum  = 0.     )

        if cut_bottom is not None:
            atoms = cut_bottom_slab(atoms      = atoms     ,
                                    cut_bottom = cut_bottom,
                                    vacuum     = 0.        )

        if symmetry == 'asymmetric':
            atoms = break_symmetry(atoms = atoms)
        elif symmetry == 'inversion':
            atoms = inversion_symmetry(atoms = atoms, vacuum = vacuum)
        elif symmetry in ('planar', 'symmetric', None):
            pass
        else:
            raise NameError('Wrong symmetry keyword')

        if layers_fixed is not None:
            atoms = fix_atoms(atoms         = atoms        , 
                              layers_fixed  = layers_fixed ,
                              layers        = layers       ,
                              symmetry      = symmetry     )

        if vacuum is not None:
            atoms.center(vacuum = 0., axis = 2)

        slab_height = atoms.cell[2][2]

        #atoms = cut_surface(atoms = atoms)

        if vacancies:
            if vacuum is not None:
                atoms.center(vacuum = vacuum/2., axis = 2)
            if type(vacancies) is not list:
                vacancies = [vacancies]
            for vacancy in vacancies:
                atoms = add_vacancy(atoms      = atoms     ,
                                    vacancy    = vacancy   ,
                                    symmetry   = symmetry  ,
                                    dimensions = dimensions,
                                    vacuum     = vacuum    )

        if adsorbates:
            if vacuum is not None:
                atoms.center(vacuum = vacuum/2., axis = 2)
            if type(adsorbates) is not list:
                adsorbates = [adsorbates]
            for adsorbate in adsorbates:
                atoms = add_adsorbate(atoms        = atoms         ,
                                      adsorbate    = adsorbate     ,
                                      symmetry     = symmetry      ,
                                      dimensions   = dimensions    ,
                                      bulk_type    = bulk.bulk_type,
                                      miller_index = miller_index  ,
                                      slab_height  = slab_height   ,
                                      vacuum       = vacuum        )

        if vacuum is not None:
            atoms.center(vacuum = vacuum/2., axis = 2)

        if sort_atoms is True:
            atoms = sort_slab(atoms = atoms)

        if scale_kpts is not None:
            kpts = calculate_kpts(atoms,
                                  cell       = bulk.atoms.cell,
                                  kpts       = bulk.kpts_bulk ,
                                  scale_kpts = scale_kpts     )
        else:
            kpts = bulk.kpts_bulk

        koffset = (0, 0, 0)

        atoms.set_pbc(True)
        self.atoms           = atoms
        self.slab_height     = slab_height
        self.constraints     = atoms.constraints
        self.kpts            = kpts
        self.koffset         = koffset
        self.bulk            = bulk
        self.input_slab      = input_slab
        self.miller_index    = miller_index
        self.surface_vectors = surface_vectors
        self.dimensions      = dimensions
        self.layers          = layers
        self.layers_fixed    = layers_fixed
        self.symmetry        = symmetry
        self.rotation_angle  = rotation_angle
        self.cut_top         = cut_top
        self.cut_bottom      = cut_bottom
        self.adsorbates      = adsorbates
        self.vacancies       = vacancies
        self.scale_kpts      = scale_kpts
        self.vacuum          = vacuum
        self.sort_atoms      = sort_atoms

    # -------------------------------------------------------------------
    #  CUT SLAB
    # -------------------------------------------------------------------

    def cut_slab(self, surface_vectors, big_dim = None,
                 origin = [0., 0.], epsi = 1e-5):
    
        self.atoms = cut_surface(atoms           = self.atoms     ,
                                 surface_vectors = surface_vectors,
                                 big_dim         = big_dim        ,
                                 origin          = origin         ,
                                 epsi            = epsi           )

    # -------------------------------------------------------------------
    #  ROTATE SLAB
    # -------------------------------------------------------------------

    def rotate_slab(self, rotation_angle):
    
        self.atoms = rotate_slab(self.atoms, rotation_angle)

    # -------------------------------------------------------------------
    #  CUT TOP SLAB
    # -------------------------------------------------------------------

    def cut_top_slab(self, cut_top, starting = 'from slab bottom',
                     vacuum = None, epsi = 1e-3):
    
        self.atoms = cut_top_slab(atoms    = self.atoms,
                                  cut_top  = cut_top   ,
                                  starting = starting  ,
                                  vacuum   = vacuum    ,
                                  epsi     = epsi      )

    # -------------------------------------------------------------------
    #  CUT BOTTOM SLAB
    # -------------------------------------------------------------------

    def cut_bottom_slab(self, cut_bottom, starting = 'from slab bottom',
                        vacuum = None, epsi = 1e-3):
    
        self.atoms = cut_bottom_slab(atoms      = self.atoms, 
                                     cut_bottom = cut_bottom, 
                                     starting   = starting  ,
                                     vacuum     = vacuum    , 
                                     epsi       = epsi      )

    # -------------------------------------------------------------------
    #  FIX ATOMS
    # -------------------------------------------------------------------

    def fix_atoms_slab(self, layers_fixed = None, layers = None, 
                       symmetry = None, vacuum = None, distance = None,
                       epsi = 1e-3):

        if layers is None:
            layers = self.layers

        if layers_fixed is None:
            layers_fixed = self.layers_fixed

        if symmetry is None:
            symmetry = self.symmetry

        if vacuum is None:
            vacuum = self.vacuum

        self.atoms = fix_atoms(atoms        = self.atoms  , 
                               layers_fixed = layers_fixed, 
                               layers       = layers      ,
                               symmetry     = symmetry    ,
                               distance     = distance    ,
                               vacuum       = vacuum      ,
                               epsi         = epsi        )

    # -------------------------------------------------------------------
    #  ADD ADSORBATES
    # -------------------------------------------------------------------

    def add_adsorbates(self, adsorbates, symmetry = None):

        if symmetry is None:
            symmetry = self.symmetry

        if type(adsorbates) is not list:
            adsorbates = [adsorbates]

        for adsorbate in adsorbates:
            self.atoms = add_adsorbate(atoms        = self.atoms         , 
                                       adsorbate    = adsorbate          , 
                                       symmetry     = symmetry           , 
                                       dimensions   = self.dimensions    , 
                                       bulk_type    = self.bulk.bulk_type,
                                       miller_index = self.miller_index  , 
                                       slab_height  = self.slab_height   ,
                                       vacuum       = self.vacuum        )

    # -------------------------------------------------------------------
    #  ADD VACANCIES
    # -------------------------------------------------------------------

    def add_vacancies(self, vacancies, symmetry = None, epsi = 1e-3):

        if symmetry is None:
            symmetry = self.symmetry

        if type(vacancies) is not list:
            vacancies = [vacancies]

        for vacancy in vacancies:
            self.atoms = add_vacancy(atoms      = self.atoms     ,
                                     vacancy    = vacancy        ,
                                     symmetry   = symmetry       ,
                                     dimensions = self.dimensions,
                                     vacuum     = self.vacuum    ,
                                     epsi       = epsi           )

    # -------------------------------------------------------------------
    #  INVESTION SYMMETRY
    # -------------------------------------------------------------------

    def inversion_symmetry_slab(self):
    
        self.atoms = inversion_symmetry(self.atoms)

    # -------------------------------------------------------------------
    #  INVESTION SYMMETRY
    # -------------------------------------------------------------------

    def check_inversion_symmetry(self,
                                 base_boundary  = False,
                                 outer_boundary = True ):
    
        sym = check_inversion_symmetry(atoms          = self.atoms    ,
                                       base_boundary  = base_boundary ,
                                       outer_boundary = outer_boundary)
    
        print('\nINVERSION SYMMETRY = {}\n'.format(sym))

    # -------------------------------------------------------------------
    #  SORT SLAB
    # -------------------------------------------------------------------

    def sort_slab(self):
    
        self.atoms = sort_slab(self.atoms)

    # -------------------------------------------------------------------
    #  ADD VACUUM
    # -------------------------------------------------------------------

    def add_vacuum(self, vacuum):
        
        self.atoms.center(vacuum = vacuum/2., axis = 2)

    # -------------------------------------------------------------------
    #  GET LAYERS HEIGHT
    # -------------------------------------------------------------------

    def get_layers_height(self):

        heights = []

        for layers in (1, 2):

            slab_copy = Slab(bulk            = self.bulk        ,
                             miller_index    = self.miller_index,
                             layers          = layers           ,
                             vacuum          = 0.               )

            heights += [slab_copy.atoms.cell[2][2]]

        layers_height = heights[1]-heights[0]

        return layers_height

    # -------------------------------------------------------------------
    #  GET KPTS
    # -------------------------------------------------------------------

    def get_kpts(self):

        kpts = calculate_kpts(atoms      = self.atoms          ,
                              cell       = self.bulk.atoms.cell,
                              kpts       = self.bulk.kpts_bulk ,
                              scale_kpts = self.scale_kpts     )

        self.kpts = kpts

        return kpts

    # -------------------------------------------------------------------
    #  UPDATE
    # -------------------------------------------------------------------

    def update(self):

        slab = Slab(bulk            = self.bulk           ,
                    input_slab      = self.input_slab     ,
                    miller_index    = self.miller_index   ,
                    surface_vectors = self.surface_vectors,
                    dimensions      = self.dimensions     ,
                    layers          = self.layers         ,
                    layers_fixed    = self.layers_fixed   ,
                    symmetry        = self.symmetry       ,
                    rotation_angle  = self.rotation_angle ,
                    cut_top         = self.cut_top        ,
                    cut_bottom      = self.cut_bottom     ,
                    adsorbates      = self.adsorbates     ,
                    vacancies       = self.vacancies      ,
                    scale_kpts      = self.scale_kpts     ,
                    vacuum          = self.vacuum         ,
                    sort_atoms      = self.sort_atoms     )

        return slab

################################################################################
# ADSORBATE CLASS
################################################################################

class Adsorbate:

    def __init__(self     ,
                 atoms    ,
                 position = None       ,
                 height   = None       ,
                 distance = None       ,
                 units    = 'unit cell',
                 site     = None       ,
                 variant  = 0          ,
                 quadrant = 0          ,
                 rot_vect = None       ,
                 angle    = 0.         ):

        self.atoms    = atoms
        self.position = position
        self.height   = height
        self.distance = distance
        self.units    = units
        self.site     = site
        self.variant  = variant
        self.quadrant = quadrant
        self.rot_vect = rot_vect
        self.angle    = angle

################################################################################
# VACANCY CLASS
################################################################################

class Vacancy:

    def __init__(self     ,
                 position ,
                 height   = None       ,
                 distance = None       ,
                 units    = 'unit cell'):

        self.position = position
        self.height   = height
        self.distance = distance
        self.units    = units

################################################################################
# CONVERT MILLER INDEX
################################################################################

def convert_miller_index(miller_index):

    if isinstance(miller_index, str):
        miller_index = list(miller_index)
        for i in range(len(miller_index)):
            miller_index[i] = int(miller_index[i])
        miller_index = tuple(miller_index)
    
    elif isinstance(miller_index, list):
        miller_index = tuple(miller_index)

    return miller_index

################################################################################
# IMPORT BULK STRUCTURE
################################################################################

def import_bulk_structure(input_bulk):

    atoms = read(input_bulk)

    return atoms

################################################################################
# BUILD BULK STRUCTURE
################################################################################

def build_bulk_structure(bulk_type, elements, lattice_constants):

    if bulk_type == 'cubic':

        cell = [[lattice_constants[0], 0., 0.],
                [0., lattice_constants[0], 0.], 
                [0., 0., lattice_constants[0]]]

        atoms = Atoms(elements[0], scaled_positions = [[0., 0., 0.]],
                      cell = cell, pbc = True)

    elif bulk_type == 'fcc reduced':

        a_lat = lattice_constants[0] / sqrt(2.)

        cell = [[a_lat, 0., 0.],
                [a_lat/2, a_lat*sqrt(3./4), 0.],
                [a_lat/2, a_lat*sqrt(1./12), a_lat*sqrt(2./3)]]

        atoms = Atoms(elements[0], scaled_positions = [[0., 0., 0.]],
                      cell = cell, pbc = True)

    elif bulk_type in ('fcc', 'bcc', 'hcp'):

        a = lattice_constants[0]
        try: c = lattice_constants[1]
        except: c = None
        cubic = True if bulk_type in ('fcc', 'bcc') else False
        atoms = ase.build.bulk(elements[0], bulk_type, a = a,  c = c,
                               cubic = cubic)

    else:
        atoms = custom_bulk(bulk_type         = bulk_type,
                            elements          = elements, 
                            lattice_constants = lattice_constants)

    return atoms

################################################################################
# CUSTOM BULK
################################################################################

def custom_bulk(bulk_type, elements, lattice_constants):

    if bulk_type == 'corundum':

        dim_cell = (3, 3, 12)

        basis = [[ 1.,  2.,  0.], [ 2.,  1.,  0.], [ 0.,  2.,  1.],
                 [ 1.,  1.,  1.], [ 2.,  0.,  1.], [ 0.,  0.,  2.],
                 [ 2.,  1.,  2.], [ 0.,  1.,  3.], [ 1.,  0.,  3.],
                 [ 2.,  2.,  3.], [ 0.,  0.,  4.], [ 1.,  2.,  4.],
                 [ 0.,  2.,  5.], [ 1.,  1.,  5.], [ 2.,  0.,  5.],
                 [ 1.,  2.,  6.], [ 2.,  1.,  6.], [ 0.,  1.,  7.],
                 [ 1.,  0.,  7.], [ 2.,  2.,  7.], [ 0.,  0.,  8.],
                 [ 2.,  1.,  8.], [ 0.,  2.,  9.], [ 1.,  1.,  9.],
                 [ 2.,  0.,  9.], [ 0.,  0., 10.], [ 1.,  2., 10.],
                 [ 0.,  1., 11.], [ 1.,  0., 11.], [ 2.,  2., 11.]]

        elem_basis = (0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
                      0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1)

        bulk_str = 'hexagonal'

    elif bulk_type == 'rutile':

        dim_cell = (6, 6, 2)

        basis = [[0.00, 0.00, 0.00], [1.85, 4.15, 0.00],
                 [4.15, 1.85, 0.00], [1.15, 1.15, 1.00],
                 [3.00, 3.00, 1.00], [4.85, 4.85, 1.00]]

        elem_basis = (0, 1, 1, 1, 0, 1)

        bulk_str = 'tetragonal'

    elif bulk_type == 'graphene':

        dim_cell = (3, 3, 2)

        basis = [[0, 0, 0], [1, 2, 0],
                 [0, 0, 1], [2, 1, 1]]

        elem_basis = (0, 0, 0, 0)

        bulk_str = 'hexagonal'

    elif bulk_type == 'rocksalt':
        
        dim_cell = (1, 1, 1)

        basis = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0],
                 [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5],
                 [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]]

        elem_basis = (0, 1, 1, 0, 1, 0, 0, 1)

        bulk_str = 'cubic'
        lattice_constants = lattice_constants[0]

    else:
        raise NameError('bulk type "{}" not implemented'.format(bulk_type))

    bravais = np.zeros((len(basis), 3))
    for i in range(len(basis)):
        for j in range(3):
            bravais[i][j] = basis[i][j]/float(dim_cell[j])

    if bulk_str == 'cubic':
        from ase.lattice.cubic import SimpleCubicFactory
        class CustomCellFactory(SimpleCubicFactory):
            bravais_basis = bravais
            element_basis = elem_basis

    elif bulk_str == 'hexagonal':
        from ase.lattice.hexagonal import HexagonalFactory
        class CustomCellFactory(HexagonalFactory):
            bravais_basis = bravais
            element_basis = elem_basis

    elif bulk_str == 'tetragonal':
        from ase.lattice.triclinic import TriclinicFactory
        class CustomCellFactory(SimpleTetragonalFactory):
            bravais_basis = bravais
            element_basis = elem_basis

    elif bulk_str == 'triclinic':
        from ase.lattice.tetragonal import SimpleTetragonalFactory
        class CustomCellFactory(TriclinicFactory):
            bravais_basis = bravais
            element_basis = elem_basis

    CustomCell = CustomCellFactory()
    atoms = CustomCell(symbol          = elements[:max(elem_basis)+1],
                       latticeconstant = lattice_constants           ,
                       size            = (1, 1, 1)                   )

    return atoms

################################################################################
# IMPORT SLAB STRUCTURE
################################################################################

def import_slab_structure(input_slab, dimensions, vacuum = None):

    atoms = read(input_slab)

    if vacuum is not None:
        atoms.center(vacuum = 0., axis = 2)

    atoms *= (dimensions[0], dimensions[1], 1)

    return atoms

################################################################################
# BUILD SLAB STRUCTURE
################################################################################

def build_slab_structure(atoms, bulk_type, elements, lattice_constants,
                         miller_index, dimensions, layers):

    if miller_index == '100':
        if bulk_type == 'fcc':
            atoms = ase.build.fcc100(elements[0],
                        size   = (dimensions[0], dimensions[1], layers),
                        a      = lattice_constants[0],
                        vacuum = 0.)
        elif bulk_type == 'bcc':
            atoms = ase.build.bcc100(elements[0],
                        size   = (dimensions[0], dimensions[1], layers),
                        a      = lattice_constants[0],
                        vacuum = 0.)

    elif miller_index == '110':
        if bulk_type == 'fcc':
            atoms = ase.build.fcc110(elements[0],
                        size   = (dimensions[0], dimensions[1], layers),
                        a      = lattice_constants[0],
                        vacuum = 0.)
        elif bulk_type == 'bcc':
            atoms = ase.build.bcc110(elements[0],
                        size   = (dimensions[0], dimensions[1], layers),
                        a      = lattice_constants[0],
                        vacuum = 0.)

    elif miller_index == '111':
        if bulk_type == 'fcc':
            atoms = ase.build.fcc111(elements[0],
                        size   = (dimensions[0], dimensions[1], layers),
                        a      = lattice_constants[0],
                        vacuum = 0.)
        elif bulk_type == 'bcc':
            atoms = ase.build.bcc111(elements[0],
                        size   = (dimensions[0], dimensions[1], layers),
                        a      = lattice_constants[0],
                        vacuum = 0.)

    elif miller_index == '0001' and bulk_type == 'hcp':
        atoms = ase.build.hcp0001(elements[0],
                    size   = (dimensions[0], dimensions[1], layers),
                    a      = lattice_constants[0], 
                    c      = lattice_constants[1],
                    vacuum = 0.)

    elif miller_index == '211' and bulk_type == 'fcc':
        layers = int(ceil(layers*3/2))
        miller_index = convert_miller_index(miller_index)
        atoms = build_slab_structure(atoms             = atoms,
                                     bulk_type         = bulk_type,
                                     elements          = elements,
                                     lattice_constants = lattice_constants,
                                     miller_index      = miller_index,
                                     dimensions        = (1, 1),
                                     layers            = layers)
        if layers % 3:
            atoms = cut_bottom_slab(atoms      = atoms,
                                    cut_bottom = 1e-3,
                                    starting   = 'from slab bottom',
                                    verbosity  = 'low')
        surface_vectors = automatic_vectors(bulk_type    = bulk_type,
                                            miller_index = miller_index)
        atoms = cut_surface(atoms           = atoms, 
                            surface_vectors = surface_vectors)
        atoms *= (dimensions[0], dimensions[1], 1)

    elif miller_index == '210' and bulk_type == 'fcc':
        layers = int(ceil(layers*3/2))
        miller_index = convert_miller_index(miller_index)
        atoms = build_slab_structure(atoms             = atoms,
                                     bulk_type         = bulk_type,
                                     elements          = elements,
                                     lattice_constants = lattice_constants,
                                     miller_index      = miller_index,
                                     dimensions        = (1, 1),
                                     layers            = layers)
        if layers % 2:
            atoms = cut_bottom_slab(atoms      = atoms,
                                    cut_bottom = 1e-3,
                                    starting   = 'from slab bottom',
                                    verbosity  = 'low')

        surface_vectors = automatic_vectors(bulk_type    = bulk_type,
                                            miller_index = miller_index)
        atoms = cut_surface(atoms           = atoms, 
                            surface_vectors = surface_vectors)
        atoms *= (dimensions[0], dimensions[1], 1)

    elif miller_index == '311' and bulk_type == 'fcc':
        layers *= 2
        miller_index = convert_miller_index(miller_index)
        atoms = build_slab_structure(atoms             = atoms,
                                     bulk_type         = bulk_type,
                                     elements          = elements,
                                     lattice_constants = lattice_constants,
                                     miller_index      = miller_index,
                                     dimensions        = (1, 1),
                                     layers            = layers)

        surface_vectors = automatic_vectors(bulk_type    = bulk_type,
                                            miller_index = miller_index)
        atoms = cut_surface(atoms           = atoms, 
                            surface_vectors = surface_vectors)
        atoms *= (dimensions[0], dimensions[1], 1)

    elif miller_index == '321' and bulk_type == 'fcc':
        layers = layers*2+1
        miller_index = convert_miller_index(miller_index)
        atoms = build_slab_structure(atoms             = atoms,
                                     bulk_type         = bulk_type,
                                     elements          = elements,
                                     lattice_constants = lattice_constants,
                                     miller_index      = miller_index,
                                     dimensions        = (1, 1),
                                     layers            = layers)

        surface_vectors = automatic_vectors(bulk_type    = bulk_type,
                                            miller_index = miller_index)
        atoms = cut_surface(atoms           = atoms, 
                            surface_vectors = surface_vectors)
        atoms *= (dimensions[0], dimensions[1], 1)

    elif miller_index == '331' and bulk_type == 'fcc':
        layers *= 3
        miller_index = convert_miller_index(miller_index)
        atoms = build_slab_structure(atoms             = atoms,
                                     bulk_type         = bulk_type,
                                     elements          = elements,
                                     lattice_constants = lattice_constants,
                                     miller_index      = miller_index,
                                     dimensions        = (1, 1),
                                     layers            = layers)

        surface_vectors = automatic_vectors(bulk_type    = bulk_type,
                                            miller_index = miller_index)
        atoms = cut_surface(atoms           = atoms, 
                            surface_vectors = surface_vectors)
        atoms *= (dimensions[0], dimensions[1], 1)

    else:
        miller_index = convert_miller_index(miller_index)
        atoms = ase.build.surface(atoms, miller_index, layers)
        for a in atoms:
            a.position = -a.position
        atoms.translate((atoms.cell[0][0]+atoms.cell[1][0],
                         atoms.cell[0][1]+atoms.cell[1][1], 0.))
        atoms.center(vacuum = 0., axis = 2)
        atoms *= (dimensions[0], dimensions[1], 1)

    return atoms

################################################################################
# AUTOMATIC VECTORS
################################################################################

def automatic_vectors(bulk_type, miller_index):

    if bulk_type == 'fcc':
        if miller_index == (1, 0, 0):
            surface_vectors = [[+0.50, +0.50], [-0.50, +0.50]]
        elif miller_index == (1, 1, 0):
            surface_vectors = [[+0.50, +0.00], [+0.00, +1.00]]
        elif miller_index == (1, 1, 1):
            surface_vectors = [[+0.50, +0.00], [+0.50, +0.50]]
        elif miller_index == (2, 1, 0):
            surface_vectors = [[+0.50, +0.50], [+0.00, +1.00]]
        elif miller_index == (2, 1, 1):
            surface_vectors = [[+1.00, -1.00], [+0.00, +0.50]]
            #surface_vectors = [[+1.00, +1.00], [+0.00, +0.50]]
        elif miller_index == (2, 2, 1):
            surface_vectors = [[+1.00, +0.00], [-0.50, +0.50]]
        elif miller_index == (3, 1, 0):
            surface_vectors = [[+0.50, +0.00], [+0.00, +1.00]]
        elif miller_index == (3, 1, 1):
            surface_vectors = [[+0.50, -0.50], [+0.00, +0.50]]
            #surface_vectors = [[+1.00, +0.00], [+0.00, +1.00]]
        elif miller_index == (3, 2, 0):
            surface_vectors = [[+0.50, +0.50], [+0.00, +1.00]]
        elif miller_index == (3, 2, 1):
            surface_vectors = [[+0.50, +0.00], [-0.50, +1.00]]
        elif miller_index == (3, 3, 1):
            surface_vectors = [[+0.50, +0.00], [-0.50, +0.50]]
        else:
            surface_vectors = [[+1.00, +0.00], [+0.00, +1.00]]

    elif bulk_type == 'corundum':
        if miller_index == (0, 0, 1):
            surface_vectors = [[+1.00, +0.00], [+0.00, +1.00]]
        if miller_index == (1, -1, 2):
            surface_vectors = [[+1.00, +0.00], [-1./3, +1./3]]
        else:
            surface_vectors = [[+1.00, +0.00], [+0.00, +1.00]]

    else:
        surface_vectors = [[+1.00, +0.00], [+0.00, +1.00]] # ( TODO: complete )

    return surface_vectors

################################################################################
# CUT SURFACE
################################################################################

def cut_surface(atoms, surface_vectors = [[1., 0.], [0., 1.]],
                big_dim = None, origin = (0., 0.), epsi = 1e-4):

    vector_a, vector_b = surface_vectors

    if big_dim is None:
        big_dim = int(ceil(max(vector_a[0], vector_a[1], 
                               vector_b[0], vector_b[1])) * 4)

    unit = np.array([[atoms.cell[0][0], atoms.cell[0][1]], 
                     [atoms.cell[1][0], atoms.cell[1][1]]])

    base = np.vstack([atoms.cell[:2][0][:2], atoms.cell[:2][1][:2]])

    atoms *= (big_dim, big_dim, 1)
    atoms.translate((-big_dim/2.*sum(base)[0]-origin[0],
                     -big_dim/2.*sum(base)[1]-origin[1], 0.))

    cell = np.array([[vector_a[0]*unit[0][0]+vector_a[1]*unit[1][0],
                      vector_a[1]*unit[1][1]],
                     [vector_b[0]*unit[0][0]+vector_b[1]*unit[1][0],
                      vector_b[1]*unit[1][1]]])

    new = np.vstack([np.dot(vector_a, base),
                     np.dot(vector_b, base)])

    del atoms [[ a.index for a in atoms
        if a.position[1] < new[0][1]/new[0][0]*a.position[0]-epsi
        or a.position[1] > new[0][1]/new[0][0]*(a.position[0]-new[1][0])
                            + cell[1][1]-epsi
        or a.position[0] < new[1][0]/new[1][1]*a.position[1]-epsi
        or a.position[0] > new[1][0]/new[1][1]*(a.position[1]-new[0][1])
                            + new[0][0]-epsi ]]

    atoms.set_cell(np.matrix([[new[0][0], new[0][1], 0.],
                              [new[1][0], new[1][1], 0.],
                              [0., 0., atoms.cell[2][2]]]))

    return atoms

################################################################################
# ROTATE SLAB
################################################################################

def rotate_slab(atoms, rotation_angle):

    if rotation_angle == 'automatic':
        rotation_angle = -atan(atoms.cell[0][1]/atoms.cell[0][0])*180/pi
        atoms.rotate(rotation_angle, v = 'z', rotate_cell = True)

    elif rotation_angle == 'invert axis':
        atoms = rotate_slab(atoms, 'automatic')
        rotation_angle = 90 + atan(atoms.cell[1][0]/atoms.cell[1][1])*180/pi
        atoms.rotate(rotation_angle, v = 'z', rotate_cell = True)
        #atoms.set_pbc(False)
        for a in atoms:
            a.position[0] = -a.position[0]
        cell = np.array([[-atoms.cell[1][0],            0.,               0.],
                         [-atoms.cell[0][0], atoms.cell[0][1],            0.],
                         [               0.,            0., atoms.cell[2][2]]])
        #atoms.set_pbc(True)
        atoms.set_cell(cell, scale_atoms = False)
        atoms = cut_surface(atoms)
        atoms = rotate_slab(atoms, 'automatic')
    else:
        atoms.rotate(rotation_angle, v = 'z', rotate_cell = True)

    return atoms

################################################################################
# FIX ATOMS
################################################################################

def fix_atoms(atoms, layers_fixed, layers, symmetry = None, distance = None,
              vacuum = None, epsi = 1e-3):

    if vacuum is not None:
        atoms.center(vacuum = 0., axis = 2)

    atoms = sort_slab(atoms)

    if symmetry in (None, 'asymmetric'):
        
        if distance is not None:

            indices = [ a.index for a in atoms 
                        if a.position[2] < atoms.cell[2][2]-distance ]
            
        else:

            indices = [ a.index for a in atoms
                        if a.index+1 <= layers_fixed/layers*len(atoms)+epsi ]

    elif symmetry in ('planar', 'symmetric', 'inversion'):
        
        if distance is not None:
            
            indices = [ a.index for a in atoms 
                        if a.position[2] > 0.+distance
                        and a.position[2] < atoms.cell[2][2]-distance ]
            
        else:

            indices = [ a.index for a in atoms
                        if a.index+1 <= 0.5*(len(atoms)+1) + 
                        0.5*len(atoms)*layers_fixed/layers+epsi
                        and a.index+1 >= 0.5*(len(atoms)+1) - 
                        0.5*len(atoms)*layers_fixed/layers-epsi ]

    atoms.set_constraint(FixAtoms(indices = indices))

    if vacuum is not None:
        atoms.center(vacuum = vacuum/2., axis = 2)

    return atoms

################################################################################
# SORT SLAB
################################################################################

def sort_slab(atoms):

    args = np.argsort(atoms.positions[:, 2])
    atoms = atoms[args]

    return atoms

################################################################################
# CUT TOP SLAB
################################################################################

def cut_top_slab(atoms, cut_top, starting = 'from slab top',
                 vacuum = None, epsi = 1e-3, verbosity = 'high'):

    if starting in ('from slab bottom', 'from slab top'):
        if vacuum is None:
            cell_height = atoms.cell[2][2]
            atoms.center(vacuum = 0., axis = 2)
            vacuum = cell_height-atoms.cell[2][2]
        else:
            atoms.center(vacuum = 0., axis = 2)

    if starting in ('from slab bottom', 'from cell bottom'):
        cut_height = cut_top+epsi
    elif starting in ('from slab top', 'from cell top'):
        cut_height = atoms.cell[2][2]-cut_top+epsi

    if verbosity == 'high':
        print('deleted top atoms:', len([ a.index for a in atoms
                                          if a.position[2] > cut_height ]))
    del atoms [[ a.index for a in atoms if a.position[2] > cut_height ]]

    if vacuum:
        atoms.center(vacuum = vacuum/2., axis = 2)

    return atoms

################################################################################
# CUT BOTTOM SLAB
################################################################################

def cut_bottom_slab(atoms, cut_bottom, starting = 'from slab bottom',
                    vacuum = None, epsi = 1e-3, verbosity = 'high'):

    if starting in ('from slab bottom', 'from slab top'):
        if vacuum is None:
            cell_height = atoms.cell[2][2]
            atoms.center(vacuum = 0., axis = 2)
            vacuum = cell_height-atoms.cell[2][2]
        else:
            atoms.center(vacuum = 0., axis = 2)

    if starting in ('from slab bottom', 'from cell bottom'):
        cut_height = cut_bottom-epsi
    elif starting in ('from slab top', 'from cell top'):
        cut_height = atoms.cell[2][2]-cut_bottom-epsi

    if verbosity == 'high':
        print('deleted bottom atoms:', len([ a.index for a in atoms
                                             if a.position[2] < cut_height ]))
    del atoms [[ a.index for a in atoms if a.position[2] < cut_height ]]

    if vacuum:
        atoms.center(vacuum = vacuum/2., axis = 2)

    return atoms

################################################################################
# BREAK SYMMETRY
################################################################################

def break_symmetry(atoms, translation = 1e-3, epsi = 1e-4):

    for a in [ a for a in atoms if a.position[2] > atoms.cell[2][2]-epsi ]:
        a.position[2] = a.position[2]+translation

    return atoms

################################################################################
# BOUNDARY ATOMS
################################################################################

def boundary_atoms(atoms, base_boundary = False, outer_boundary = True, 
                   epsi = 1e-3):

    atoms_plus = cp.deepcopy(atoms)

    atoms_plus = cut_surface(atoms_plus)

    for a in atoms:
        if abs(a.position[0]) < epsi and abs(a.position[1]) < epsi:
            a_plus = cp.deepcopy(a)
            a_plus.position[:2] += sum(atoms.cell[:2])[:2]
            atoms_plus += a_plus

        if (abs(a.position[0] - a.position[1] * atoms.cell[1][0] /
            atoms.cell[1][1]) < epsi):
            a_plus = cp.deepcopy(a)
            a_plus.position[:2] += atoms.cell[0][:2]
            atoms_plus += a_plus

        if (abs(a.position[1] - a.position[0] * atoms.cell[0][1] /
            atoms.cell[0][0]) < epsi):
            a_plus = cp.deepcopy(a)
            a_plus.position[:2] += atoms.cell[1][:2]
            atoms_plus += a_plus

    if base_boundary is True:
        for a in atoms:
            if abs(a.position[2]) < epsi:
                a_plus = cp.deepcopy(a)
                a_plus.position[2] += atoms.cell[2][2]
                atoms_plus += a_plus

    if outer_boundary is True:
        for a in atoms:
            if (abs(a.position[0] - atoms.cell[0][0] - atoms.cell[1][0]) <
                epsi and abs(a.position[1] - atoms.cell[0][1] -
                atoms.cell[1][1]) < epsi):
                a_plus = cp.deepcopy(a)
                a_plus.position[:2] -= sum(atoms.cell[:2])[:2]
                atoms_plus += a_plus

            if (abs(a.position[0] - atoms.cell[0][0] - a.position[1] *
                atoms.cell[1][0] / atoms.cell[1][1]) < epsi):
                a_plus = cp.deepcopy(a)
                a_plus.position[:2] -= atoms.cell[0][:2]
                atoms_plus += a_plus

            if (abs(a.position[1] - atoms.cell[1][1] - a.position[0] *
                atoms.cell[0][1] / atoms.cell[0][0]) < epsi):
                a_plus = cp.deepcopy(a)
                a_plus.position[:2] -= atoms.cell[1][:2]
                atoms_plus += a_plus

    return atoms_plus

################################################################################
# INVERSION SYMMETRY
################################################################################

def inversion_symmetry(atoms, vacuum = None):

    if vacuum is None:
        base_boundary = True
    else:
        atoms.center(vacuum = 0., axis = 2)
        base_boundary = False

    atoms = cut_surface(atoms)

    sym = check_inversion_symmetry(atoms          = atoms        ,
                                   base_boundary  = base_boundary,
                                   outer_boundary = True         )

    print('\n INVERSION SYMMETRY =', sym)

    if sym is not True:

        atoms_new = create_inversion_symmetry(
                        atoms          = cp.deepcopy(atoms), 
                        base_boundary  = base_boundary     , 
                        outer_boundary = True              ,
                        cut_slab       = False             )

        sym = check_inversion_symmetry(atoms          = atoms_new    ,
                                       base_boundary  = base_boundary,
                                       outer_boundary = True         )

        print('\n INVERSION SYMMETRY =', sym)

        if sym is True:

            atoms = atoms_new

        else:

            #print('\nBIG CELL INVERSION SYMMETRY')
            #
            #atoms *= (2, 2, 1)
            #
            #atoms_new = create_inversion_symmetry(
            #                atoms          = cp.deepcopy(atoms), 
            #                base_boundary  = base_boundary     , 
            #                outer_boundary = True              ,
            #                cut_slab       = False             )
            #
            #sym = check_inversion_symmetry(atoms          = atoms_new    ,
            #                               base_boundary  = base_boundary,
            #                               outer_boundary = True         )

            if sym is True:

                atoms = atoms_new

            else:

                atoms = create_inversion_symmetry(
                            atoms          = atoms        ,
                            base_boundary  = base_boundary, 
                            outer_boundary = True         ,
                            cut_slab       = True         )

                sym = check_inversion_symmetry(atoms          = atoms        ,
                                               base_boundary  = base_boundary,
                                               outer_boundary = True         )
            
            #atoms = cut_surface(atoms           = atoms                 ,
            #                    surface_vectors = [[0.5, 0.], [0., 0.5]])

            sym = check_inversion_symmetry(atoms          = atoms        ,
                                           base_boundary  = base_boundary,
                                           outer_boundary = True         )

            print('\n INVERSION SYMMETRY =', sym)

            if sym is not True:
                print('NO SYMMETRY FOUND!')

    if vacuum:
        atoms.center(vacuum = vacuum/2., axis = 2)

    return atoms

################################################################################
# CREATE INVERSION SYMMETRY
################################################################################

def create_inversion_symmetry(atoms, 
                              base_boundary  = False, 
                              outer_boundary = False,
                              cut_slab       = False,
                              epsi           = 1e-3 ):

    print('\nCREATING INVERSION SYMMETRY')

    atoms_plus = boundary_atoms(atoms          = atoms         ,
                                base_boundary  = base_boundary ,
                                outer_boundary = outer_boundary)

    centre = find_inversion_centre(atoms          = atoms_plus    ,
                                   cut_slab       = cut_slab      ,
                                   base_boundary  = base_boundary ,
                                   outer_boundary = outer_boundary)

    origin = centre-sum(atoms.cell)/2.

    if cut_slab is True:

        if origin[2] < 0.:
            print('deleted top atoms:', len([ a.index for a in atoms
                if a.position[2] > 2.*centre[2]+epsi ]))
            del atoms [[ a.index for a in atoms
                if a.position[2] > 2.*centre[2]+epsi ]]
        elif origin[2] > 0.:
            print('deleted bottom atoms:', len([ a.index for a in atoms
                if a.position[2] < 2.*centre[2]-atoms.cell[2][2]-epsi ]))
            del atoms [[ a.index for a in atoms
                if a.position[2] < 2.*centre[2]-atoms.cell[2][2]-epsi ]]

    atoms = cut_surface(atoms, origin = origin[:2])

    return atoms

################################################################################
# FIND INVERSION CENTRE
################################################################################

def find_inversion_centre(atoms,
                          cut_slab       = False,
                          base_boundary  = False,
                          outer_boundary = False,
                          epsi           = 1e-3 ):

    centre = [0., 0., 0.]

    c_matrix = np.array([(a.position + b.position)/2.
                         for a in atoms
                         for b in atoms 
                         if b.symbol == a.symbol
                         if b.magmom == a.magmom
                         if b.index >= a.index ])

    print('number of centres =', len(c_matrix))

    indices = np.array([len([j for j in range(i, len(c_matrix)) 
                             if np.allclose(c_matrix[i], c_matrix[j],
                                            rtol = 1e-1, atol = 1e-2)])
                        for i in range(len(c_matrix))])

    middles = np.copy(indices)

    for i in range(len(middles)):
        if abs(c_matrix[i][2]-atoms.cell[2][2]/2.) > 1e-2:
            middles[i] = 0

    n_mid = len([i for i in range(len(middles)) if middles[i] > 1])

    print('number of centres in the middle =', n_mid)

    if cut_slab is False:

        for i in range(len(middles)):

            if middles[i] > 1:

                centre = c_matrix[i]
                origin = centre-sum(atoms.cell)/2.

                atoms_new = cut_surface(atoms  = cp.deepcopy(atoms),
                                        origin = origin[:2]        )

                sym = check_inversion_symmetry(atoms          = atoms_new     ,
                                               base_boundary  = base_boundary ,
                                               outer_boundary = outer_boundary)

                if sym is True:
                    break

    else:

        del_vector = np.ones(len(c_matrix))*len(atoms)

        for i in range(len(c_matrix)):

            if indices[i] > 1:

                centre = c_matrix[i]
                origin = centre-sum(atoms.cell)/2.
            
                atoms_new = cut_surface(atoms  = cp.deepcopy(atoms),
                                        origin = origin[:2]        )
            
                if origin[2] < 0.:
    
                    height = 2.*centre[2]+epsi
                    del_atoms = len([ a.index for a in atoms_new
                                      if a.position[2] > height ])
                    del atoms_new [[ a.index for a in atoms_new
                                      if a.position[2] > height ]]
    
                elif origin[2] > 0.:
    
                    height = 2.*centre[2]-atoms_new.cell[2][2]-epsi
                    del_atoms = len([ a.index for a in atoms
                                      if a.position[2] < height ])
                    del atoms_new [[ a.index for a in atoms_new
                                      if a.position[2] < height ]]
            
                sym = check_inversion_symmetry(atoms          = atoms_new     ,
                                               base_boundary  = base_boundary ,
                                               outer_boundary = outer_boundary)
            
                if sym is True:
                    del_vector[i] = del_atoms

        centre = c_matrix[np.argmin(del_vector)]

    return centre

################################################################################
# CHECK INVERSION SYMMETRY
################################################################################

def check_inversion_symmetry(atoms,
                             base_boundary  = False, 
                             outer_boundary = True ):

    inversion = False

    print('\nCHECKING INVERSION SYMMETRY')

    cont = 0

    atoms_copy = cp.deepcopy(atoms)

    atoms_copy = cut_surface(atoms_copy)

    atoms_copy = boundary_atoms(atoms          = atoms_copy    ,
                                base_boundary  = base_boundary ,
                                outer_boundary = outer_boundary)

    centre = sum(atoms_copy.cell)/2.

    for a in atoms_copy:

        a_check = 2.*centre-a.position
        equal_check = False

        for b in [b for b in atoms_copy 
                  if b.symbol == a.symbol
                  if b.magmom == a.magmom]:

            equal = np.allclose(a_check, b.position, rtol = 1e-1, atol = 1e-2)

            if equal is True:
                cont += 1
                equal_check = True
                break

    print('n atoms  = {0:4d}'.format(len(atoms_copy)))
    print('n checks = {0:4d}'.format(cont))

    if cont >= len(atoms_copy):
        inversion = True

    return inversion

################################################################################
# STANDARD ADSORBATE
################################################################################

def standard_adsorbate(adsorbate, bulk_type = None, miller_index = None):

    if bulk_type in ('fcc', 'bcc'):

        if miller_index in ('100', (1, 0, 0)):
            site = {'top':0,
                    'sbr':1, 'brg':1, 'shortbridge':1, 'bridge':1,
                    'hol':2, 'hollow':2}
            pos = [[[0.00, 0.00, +1.50]],                      # top
                   [[1./2, 0.00, +1.00]],                      # sbr
                   [[1./2, 1./2, +0.80]]]                      # hol
            
        elif miller_index in ('110', (1, 1, 0)):
            site = {'top':0,
                    'sbr':1, 'brg':1, 'shortbridge':1, 'bridge':1,
                    'lbr':2, 'longbridge':2,
                    'lho':3, 'hol':3, 'longhollow':3, 'hollow':3}
            pos = [[[0.00, 0.00, +1.50]],                      # top
                   [[0.00, 1./2, +1.00]],                      # sbr
                   [[1./2, 0.00, +1.00]],                      # lbr
                   [[1./2, 1./2, +0.80]]]                      # lho
            if bulk_type == 'bcc':
                pos[3] = [[1./3, 1./3, +0.80]]                 # lho

        elif miller_index in ('111', (1, 1, 1)):
            site = {'top':0,
                    'sbr':1, 'brg':1, 'shortbridge':1, 'bridge':1,
                    'fcc':2,
                    'hcp':3}
            pos = [[[0.00, 0.00, +1.50]],                      # top
                   [[1./2, 0.00, +1.00]],                      # sbr
                   [[1./3, 1./3, +0.80]],                      # fcc
                   [[2./3, 2./3, +0.80]]]                      # hcp

        if miller_index in ('211', (2, 1, 1)):
            site = {'top':0,
                    'sbr':1, 'brg':1, 'shortbridge':1, 'bridge':1,
                    'hol':2, 'hollow':2,
                    'fcc':3,
                    'hcp':4}
            pos = [[[0.00, 0.00, +2.00], [1./3, 0.00, +0.00],
                    [2./3, 1./2, +0.80]],                      # top
                   [[0.00, 1./2, +1.60], [0.22, 0.00, +0.00],
                    [1./3, 1./2, -0.40], [0.48, 0.24, -0.40],
                    [2./3, 0.00, +0.20], [0.80, 0.24, +0.40]], # sbr
                   [[1./6, 1./2, -0.40]],                      # hol
                   [[1./2, 0.00, -0.40]],                      # fcc
                   [[0.42, 1./2, -0.40]]]                      # hcp
    
        if miller_index in ('311', (3, 1, 1)):
            site = {'top':0,
                    'sbr':1, 'brg':1, 'shortbridge':1, 'bridge':1,
                    'hol':2, 'hollow':2,
                    'fcc':3,
                    'hcp':4}
            pos = [[[0.00, 0.00, +2.00], [0.54, 0.82, +0.80]], # top
                   [[0.00, 1./2, +1.60], [0.38, 0.52, +0.80],
                    [0.65, 0.20, +0.60]],                      # sbr
                   [[0.35, 0.05, +0.30]],                      # hol
                   [[0.68, 0.36, +0.35]],                      # fcc
                   [[0.80, 0.20, +0.40]]]                      # hcp
    
        if miller_index in ('331', (3, 3, 1)):
            site = {'top':0,
                    'sbr':1, 'brg':1, 'shortbridge':1, 'bridge':1,
                    'fcc':2,
                    'hcp':3}
            pos = [[[0.00, 0.00, +2.00], [0.63, 0.32, +1.00],
                    [1./3, 2./3, +0.40]],                      # top
                   [[0.00, 0.50, +1.50], [1./3, 0.15, +0.00],
                    [0.74, 0.62, +0.70], [0.60, 0.75, +0.50],
                    [0.25, 0.75, +0.40], [0.36, 0.48, +0.35]], # sbr
                   [[0.80, 0.40, +0.70], [0.20, 0.60, +0.40],
                    [0.43, 0.73, +0.40]],                      # fcc
                   [[0.68, 0.84, +0.70]]]                      # hcp

    quadrant_shifts = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.],
                                [0., 2.], [1., 2.], [2., 2.], [2., 1.],
                                [2., 0.], [0., 3.], [1., 3.], [2., 3.],
                                [3., 3.], [3., 2.], [3., 1.], [3., 0.]])

    adsorbate.position = pos[site[adsorbate.site]][adsorbate.variant][:2]
    
    if adsorbate.quadrant is not None:
        adsorbate.position += quadrant_shifts[adsorbate.quadrant]

    if adsorbate.distance is None and adsorbate.height is None:
        adsorbate.distance = pos[site[adsorbate.site]][adsorbate.variant][2]

    adsorbate.units = 'unit cell'

    return adsorbate

################################################################################
# ADD ADSORBATE
################################################################################

def add_adsorbate(atoms, adsorbate, symmetry = None, dimensions = (1, 1),
                  bulk_type = None, miller_index = None, slab_height = None,
                  vacuum = None, cut_surface_first = False):

    if cut_surface_first is True:
        atoms = cut_surface(atoms)

    if vacuum is None:
        vacuum = 0.

    if adsorbate.site is not None:
        adsorbate = standard_adsorbate(adsorbate, bulk_type, miller_index)

    if adsorbate.units in ('A', 'angstrom'):
        units = ((1., 0.), (0., 1.))
    elif adsorbate.units == 'slab cell':
        units = (atoms.cell[0][:2], atoms.cell[1][:2])
    elif adsorbate.units == 'unit cell':
        units = (atoms.cell[0][:2]/dimensions[0],
                 atoms.cell[1][:2]/dimensions[1])
    else:
        raise NameError('Wrong adsorbate.units keyword')

    if adsorbate.height is not None:
        adsorbate.distance = adsorbate.height-atoms.cell[2][2]+vacuum/2.
    elif adsorbate.distance is None:
        raise TypeError('Specify adsorbate height or distance attributes')

    ads = cp.deepcopy(adsorbate.atoms)

    ads.translate((0., 0., adsorbate.distance))

    if adsorbate.rot_vect is not None:
        ads.rotate(adsorbate.angle, v = adsorbate.rot_vect)

    ads.translate((np.dot(adsorbate.position, [units[0][0], units[1][0]]),
                   np.dot(adsorbate.position, [units[0][1], units[1][1]]),
                   atoms.cell[2][2]-vacuum/2.))

    atoms += ads

    if symmetry == 'planar':
    
        ads_sym = cp.deepcopy(ads)
        for a in ads_sym:
            a.position[2] = atoms.cell[2][2]-a.position[2]
        atoms += ads_sym

    elif symmetry in ('symmetric', 'inversion'):
    
        ads_sym = cp.deepcopy(ads)
        for a in ads_sym:
            a.position[0] = sum(atoms.cell[:2])[0]-a.position[0]
            a.position[1] = sum(atoms.cell[:2])[1]-a.position[1]
            a.position[2] = atoms.cell[2][2]-a.position[2]
        atoms += ads_sym

    return atoms

################################################################################
# ADD VACANCY
################################################################################

def add_vacancy(atoms, vacancy, symmetry = None, dimensions = (1, 1),
                vacuum = None, epsi = 1e-3, cut_surface_first = False):

    if cut_surface_first is True:
        atoms = cut_surface(atoms)

    if vacuum is None:
        vacuum = 0.

    if vacancy.height is not None:
        vacancy_height = vacancy.height
    elif vacancy.distance is not None:
        vacancy_height = atoms.cell[2][2]-vacuum/2.+vacancy.distance
    else:
        raise TypeError('Specify vacancy height or distance attributes')

    if vacancy.units in ('A', 'angstrom'):
        units = ((1., 0.), (0., 1.))
    elif vacancy.units == 'slab cell':
        units = (atoms.cell[0][:2], atoms.cell[1][:2])
    elif vacancy.units == 'unit cell':
        units = (atoms.cell[0][:2]/dimensions[0],
                 atoms.cell[1][:2]/dimensions[1])
    else:
        raise NameError('Wrong vacancy.units keyword')

    vacancy_pos = (np.dot(vacancy.position, [units[0][0], units[1][0]]),
                   np.dot(vacancy.position, [units[0][1], units[1][1]]),
                   vacancy_height)

    del atoms [[ a.index for a in atoms if np.allclose(vacancy_pos,
                 a.position, rtol = 1e-2, atol = epsi) ]]

    if symmetry == 'planar':

        vacancy_pos[2] = atoms.cell[2][2]-vacancy_height

        del atoms [[ a.index for a in atoms if np.allclose(vacancy_pos,
                     a.position, rtol = 1e-2, atol = epsi) ]]

    elif symmetry in ('symmetric', 'inversion'):

        vacancy_sym = [sum(atoms.cell[:2])[0]-vacancy_pos[0],
                       sum(atoms.cell[:2])[1]-vacancy_pos[1],
                       atoms.cell[2][2]-vacancy_height]

        if (abs(vacancy_pos[0]-vacancy_pos[1]*atoms.cell[1][0] /
            atoms.cell[1][1]) < epsi
        or abs(vacancy_pos[1]-vacancy_pos[0]*atoms.cell[0][1] /
               atoms.cell[0][0]) < epsi):
            vacancy_sym[:2] = vacancy_pos[:2]

        del atoms [[ a.index for a in atoms if np.allclose(vacancy_sym,
                     a.position, rtol = 1e-2, atol = epsi) ]]

    return atoms

################################################################################
# CALCULATE KPTS
################################################################################

def calculate_kpts(atoms, cell, kpts, scale_kpts = 'xy', epsi = 1e-4):

    if kpts is not None:

        kpts = list(np.copy(kpts))
        
        kpts[0] *= np.linalg.norm(cell[0])/np.linalg.norm(atoms.cell[0])
        kpts[0] = int(np.ceil(kpts[0]-epsi))
    
        kpts[1] *= np.linalg.norm(cell[1])/np.linalg.norm(atoms.cell[1])
        kpts[1] = int(np.ceil(kpts[1]-epsi))
    
        if scale_kpts == 'xyz':
            kpts[2] *= np.linalg.norm(cell[2])/np.linalg.norm(atoms.cell[2])
            kpts[2] = int(np.ceil(kpts[2]-epsi))
        else:
            kpts[2] = 1

    return kpts

################################################################################
# END
################################################################################
