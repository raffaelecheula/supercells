# supercell_builder (version 0.1.10)

## **Bulk:**

**`class Bulk(bulk_type, input_bulk, elements, lattice_constants, kpts_bulk)`**

### Parameters:

  * **`bulk_type        `** : `'fcc'` - `'bcc'` - `'hcp'` - `'cubic'` - `'corundum'` - `'rutile'` - `'graphene'` _# Name of the bulk structure._
  * **`input_bulk       `** : `None` - `'file.in'` _# If not None, the bulk structure is read from the input file 'file.in'._
  * **`elements         `** : `'element'` - `('element1', 'element2', ...)` _# A tuple containing the elements of the bulk structure._
  * **`lattice_constants`** : `a` - `(a, c)` _# Lattice constants of the bulk structure._
  * **`kpts_bulk        `** : `(kpx, kpy, kpz)` _# Grid of k-points of the bulk structure._

### Definitions:

  * `update()` _# Update the bulk class instance._


## **Slab:**

### Parameters:

**`class Slab(bulk, input_slab, miller_index, surface_vectors, dimensions, layers, layers_fixed, symmetry, rotation_angle, cut_top, cut_bottom, adsorbates, vacancies, scale_kpts, vacuum, sort_atoms)`**

  * **`bulk           `** : `bulk` _# A supercell_builder.Bulk class instance to build the slab._
  * **`input_slab     `** : `None` - `'file.in'` _# If not None, the slab structure is read from the input file 'file.in'._
  * **`miller_index   `** : `'hkl'` - `(h, k, l)` _# A string or a tuple indicating the sequence of Miller indices to create the slab cutting the bulk structure._
  * **`surface_vectors`** : `None` - `[[xa, ya], [xb, yb]]` _# A list of two surface vectors to cut the slab with vectors perpendicular to the z axis._
  * **`dimensions     `** : `(x, y)` _# Number of repetition of the unit cell in the x and y directions._
  * **`layers         `** : `layers` _# Number of layers of the slab._
  * **`layers_fixed   `** : `None` - `layers_fixed` _# Number of layers of the slab with fixed positions._
  * **`symmetry       `** : `None` - `'asymmetric'` - `'planar'` - `'inversion'` _# Type of symmetry of the slab. The slab bottom is cut to impose inversion symmetry to the slab._
  * **`rotation_angle `** : `None` - `'automatic'` - `'invert axis'` - `angle` _# Angle of rotation of the slab; 'automatic' rotates the slab until the x' surface vector is parallel to the x axis; 'invert axis' invert the x' and y' surface vectors_
  * **`cut_top        `** : `None` - `cut` _# Cut the top of the slab (length calculated from the slab bottom)._
  * **`cut_bottom     `** : `None` - `cut` _# Cut the bottom of the slab (length calculated from the slab bottom)._
  * **`adsorbates     `** : `None` - `[adsorbate1, adsorbate2, ...]` _# A list of supercell_builder.Adsorbate class instances to add to the cell, preserving symmetry._
  * **`vacancies      `** : `None` - `[vacancy1, vacancy2, ...]` _# A list of supercell_builder.Vacancy class instances to add to the cell, preserving symmetry._
  * **`scale_kpts     `** : `None` - `'xy'` - `'xyz'` _# Directions of scaling of the k-points to preserve k-points spacing with respect to the bulk structure. For the 'xy' option, 1 k-point is set in the z direction._
  * **`vacuum         `** : `None` - `vacuum` _# If not None, vacuum is added to the supercel, half on top, half on bottom._
  * **`sort_atoms     `** : `True` - `False` _# If True, the atoms are sorted with respect to their z position._

### Definitions:

  * `cut_slab(surface_vectors)` _# Cut the slab structure with the defined surface vectors parallel to the z direction._
  * `rotate_slab(rotation_angle)` _# Rotate the slab structure with the defined rotation angle._
  * `cut_top_slab(cut_top, starting = 'from slab bottom')` _# Cut the top of the slab (starting options: 'from slab bottom', 'from slab top', 'from cell bottom', 'from cell top')._
  * `cut_bottom_slab(cut_bottom, starting = 'from slab bottom')` _# Cut the bottom of the slab (starting options: 'from slab bottom', 'from slab top', 'from cell bottom', 'from cell top')._
  * `fix_atoms(layers_fixed)` _# Fix the defined number of layers of the slab._
  * `add_adsorbates(adsorbates)` _# Add adsorbates to the slab structure._
  * `add_vacancies(vacancies)` _# Create vacancies on the slab structure._
  * `sort_slab()` _# Sort the atoms of the slab structure with respect to their z position._
  * `update()` _# Update the slab class instance._

## **Adsorbate:**

**`class Adsorbate(atoms, position, distance, units, site, variant, quadrant)`**

### Parameters:

  * **`atoms   `** : `atoms` _# An ase.Atoms class instance._
  * **`position`** : `None` - `(x, y)` _# Vector position of the adsorbate with units defined in the units attribute._
  * **`height  `** : `None` - `distance` _# Height of the adsorbate._
  * **`distance`** : `None` - `distance` _# Distance from the slab top on the z direction._
  * **`units   `** : `'angstrom'` - `'slab cell'` - `'unit cell'` _# Units for defining the position vectors._
  * **`site    `** : `None` - `'top'` - `'brg'` - `'sbr'` - `'lbr'` - `'hol'` - `'lho'` - `'fcc'` - `'hcp'` _# Name of the standard site of the adsorbate._ 
  * **`variant  `** : `None` - `number` _# On high Miller index surfaces, discriminates different adsorption sites with the same name._
  * **`quadrant`** : `None` - `quadrant` _# If not None, translates the Adsorbate to diffenent neighbors unit cells._

## **Vacancy:**

**`class Vacancy(position, distance, starting, units)`**

### Parameters:

  * **`position`** : `(x, y)` _# Vector position of the vacancy with units defined in the units attribute._
  * **`height  `** : `None` - `distance` _# Height of the vacancy._
  * **`distance`** : `None` - `distance` _# Distance from the slab top on the z direction._
  * **`units   `** : `'angstrom'` - `'slab cell'` - `'unit cell'` _# Units for defining the position vectors._

## Authors:

  * Raffaele Cheula (raffaele.cheula@polimi.it)
