# supercell_builder

###################################################################
# SUPERCELL BUILDER version 0.1
# Distributed under the GPLv3 license
# Author: Raffaele Cheula
# raffaele.cheula@polimi.it
# Laboratory of Catalysis and Catalytic Processes (LCCP)
# Department of Energy, Politecnico di Milano"
###################################################################

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
