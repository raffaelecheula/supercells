# supercell_builder (version 0.1.5)

## **Bulk:**

**`class Bulk(bulk_type, input_bulk, elements, lattice_constants, kpts_bulk)`**

### Parameters:

  * **`bulk_type        `** : `'fcc'` - `'bcc'` - `'hcp'` - `'simple cubic'` - `'corundum'` - `'rutile'` - `'graphene'` _-> Name of the bulk structure._
  * **`input_bulk       `** : `None` - `'file.in'` _-> If not None, the bulk structure is read from the input file 'file.in'._
  * **`elements         `** : `'element'` - `('element1', 'element2', ...)` _-> A list of elements of the bulk structure._
  * **`lattice_constants`** : `a` - `(a, c)` _-> Lattice constants of the bulk structure._
  * **`kpts_bulk        `** : `(kpx, kpy, kpz)` _-> Grid of k points of the bulk structure._

### Definitions:

  * `update()` _# Update the bulk class instance._


## **Slab:**

**`class Slab(bulk, input_slab, miller_index, surface_vectors, dimensions, layers, layers_fixed, symmetry, rotation_angle, cut_top, cut_bottom, adsorbates, vacancies, scale_kpts, vacuum, sort_atoms)`**

  * **`bulk           `** : `bulk` _# A supercell_builder.Bulk class instance._
  * **`input_slab     `** : `None` - `'file.in'` _# If not None, the slab structure is read from the input file 'file.in'._
  * **`miller_index   `** : `'hkl'` - `(h, k, l)` _# A string or a tuple indicating the sequence of Miller indices to create the slab cutting the bulk structure._
  * **`surface_vectors`** : `None` - `[[a00, a01], [a10, a11]]` _# Surface vectors to cut the slab with vectors perpendicular to the z axis._
  * **`dimensions     `** : `(x, y)` _# Number of repetition of the unit cell in the x and y directions._
  * **`layers         `** : `layers` _# Number of layers of the slab._
  * **`layers_fixed   `** : `layers_fixed` _# Number of layers of the slab with fixed positions._
  * **`symmetry       `** : `None` - `'asymmetric'` - `'planar'` - `'inversion'` _# Type of symmetry of the slab._
  * **`rotation_angle `** : `None` - `'automatic'` - `'invert axis'` - `angle` _# Angle of rotation of the slab._
  * **`cut_top        `** : `None` - `cut` _# Cut the top of the slab (length calculated from the slab bottom)._
  * **`cut_bottom     `** : `None` - `cut` _# Cut the bottom of the slab (length calculated from the slab bottom)._
  * **`adsorbates     `** : `None` - `[adsorbate1, adsorbate2, ...]` _# A list of supercell_builder.Adsorbate class instances._
  * **`vacancies      `** : `None` - `[vacancy1, vacancy2, ...]` _# A list of supercell_builder.Vacancy class instances._
  * **`scale_kpts     `** : `None` - `'xy'` - `'xyz'` _# Directions of scaling of the k points to preserve k points spacing with respect to the bulk structure._
  * **`vacuum         `** : `None` - `vacuum` _# If not None, vacuum is added to the supercell._
  * **`sort_atoms     `** : `True` - `False` _# If True, the atoms are sorted with respect to their z position._

### Definitions:

  * `cut_slab(surface_vectors, big_dim = None, origin = [0., 0.], epsi = 1e-5)`
  * `rotate_slab(rotation_angle)`
  * `cut_top_slab(cut_top, starting = 'from slab bottom', vacuum = None, epsi = 1e-5)`
  * `cut_bottom_slab(cut_bottom, starting = 'from slab bottom', vacuum = None, epsi = 1e-5)`
  * `fix_atoms(layers_fixed, layers = None, symmetry = None)`
  * `add_adsorbates(adsorbates, symmetry = None)`
  * `add_vacancies(vacancies, symmetry = None)`
  * `sort_slab()`
  * `update()`

## **Adsorbate:**

**`class Adsorbate(atoms, position, distance, units, site, number, quadrant)`**

  * **`atoms   `** : `atoms` _# An ase.Atoms class instance._
  * **`position`** : `None` - `(x, y)` _# Position of the adsorbate with units defined in the units attribute._
  * **`distance`** : `None` - `distance` _# Distance from the slab top atom._
  * **`units   `** : `'angstrom'` - `'slab cell'` - `'unit cell'` _# Units for defining the position vectors._
  * **`site    `** : `None` - `'top'` - `'brg'` - `'sbr'` - `'lbr'` - `'hol'` - `'lho'` - `'fcc'` - `'hcp'` _# Name of the standard site of the adsorbate._ 
  * **`number  `** : `None` - `number` _# On high Miller index surfaces, discriminates different adsorption sites with the same name._
  * **`quadrant`** : `None` - `quadrant` _# If not None, translates the Adsorbate to diffenent neighbors unit cells._
