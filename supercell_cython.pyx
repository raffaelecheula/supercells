################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, raffaele.cheula@polimi.it
################################################################################

# PYTHON IMPORT

from __future__ import absolute_import, division, print_function
import numpy as np

# CYTHON IMPORT

cimport cython
cimport numpy as np
from cpython.array cimport array
#from cython.operator cimport dereference as deref
from libc.math cimport atan, acos, sqrt, round

# PARAMETERS

cdef double pi = acos(-1)

################################################################################
# MATCH SLABS DIMENSIONS CYTHON
################################################################################

@cython.cdivision(True)    # use C semantics for division
@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
def match_slabs_dimensions_cython(double [:, ::1] cell_a not None,
                                  double [:, ::1] cell_b not None,
                                  int nmax_a,
                                  int nmax_b,
                                  double stretch_max,
                                  double area_max,
                                  double toll_rotation,
                                  double epsi,
                                  double minimum_angle = pi / 5.):

    cdef:
        unsigned int i, j, ai, aj, ak, al, bi, bj, bk, bl
        int a_00, a_01, a_10, a_11, b_00, b_01, b_10, b_11
        double cell_a_00, cell_a_01, cell_a_10, cell_a_11
        double cell_b_00, cell_b_01, cell_b_10, cell_b_11
        double new_a_00, new_a_01, new_a_10, new_a_11
        double new_b_00, new_b_01, new_b_10, new_b_11
        double area, deform_nrg, angle_a, angle_b, diff_0, diff_1
        double lenght_a_0, lenght_a_1, lenght_b_0, lenght_b_1
        list vect_a, vect_b, vect_a_opt = [], vect_b_opt = []
        list areas = [], deform_nrgs = []
        bint invert
        bint match_dimensions = False
        int [::1] index_a_00_11 = np.empty((nmax_a), dtype = np.int32)
        int [::1] index_a_01_10 = np.empty((2*nmax_a+1), dtype = np.int32)
        int [::1] index_b_00_11 = np.empty((nmax_b), dtype = np.int32)
        int [::1] index_b_01_10 = np.empty((2*nmax_b+1), dtype = np.int32)
        double [:, ::1] new_a = np.empty((2, 2), dtype = np.float64)
        double [:, ::1] new_b = np.empty((2, 2), dtype = np.float64)

    assert cell_a.shape[0] == cell_a.shape[1] == 2
    assert cell_b.shape[0] == cell_b.shape[1] == 2
    cell_a_00, cell_a_01 = cell_a[0]
    cell_a_10, cell_a_11 = cell_a[1]
    cell_b_00, cell_b_01 = cell_b[0]
    cell_b_10, cell_b_11 = cell_b[1]

    for i in range(nmax_a):
        index_a_00_11[i] = i+1
    
    index_a_01_10[0] = 0
    for i in range(1, nmax_a):
        index_a_01_10[2*i-1] = i
        index_a_01_10[2*i] = -i

    for i in range(nmax_b):
        index_b_00_11[i] = i+1

    index_b_01_10[0] = 0
    for i in range(1, nmax_b):
        index_b_01_10[2*i-1] = i
        index_b_01_10[2*i] = -i

    print('ADAPTING SLABS DIMENSIONS\n')
    print('nmax slab a = {}'.format(nmax_a))
    print('nmax slab b = {}\n'.format(nmax_b))

    for ai in range(2*nmax_a+1):
        a_01 = index_a_01_10[ai]
        for aj in range(2*nmax_a+1):
            a_10 = index_a_01_10[aj]
            for ak in range(nmax_a):
                a_00 = index_a_00_11[ak]
                for al in range(nmax_a):
                    a_11 = index_a_00_11[al]

                    new_a_00 = cell_a_00 * a_00 + cell_a_10 * a_01
                    new_a_01 = cell_a_01 * a_00 + cell_a_11 * a_01
                    new_a_10 = cell_a_00 * a_10 + cell_a_10 * a_11
                    new_a_11 = cell_a_01 * a_10 + cell_a_11 * a_11

                    area = new_a_00 * new_a_11 - new_a_01 * new_a_10

                    if new_a_00 > epsi and new_a_11 > epsi and area < area_max:

                        angle_a = pi/2 - atan(new_a_01 / new_a_00) - atan(new_a_10 / new_a_11)

                        if minimum_angle < angle_a < pi - minimum_angle:

                            for bi in range(2*nmax_b+1):
                                b_01 = index_b_01_10[bi]
                                for bj in range(2*nmax_b+1):
                                    b_10 = index_b_01_10[bj]
                                    for bk in range(nmax_b):
                                        b_00 = index_b_00_11[bk]
                                        for bl in range(nmax_b):
                                            b_11 = index_b_00_11[bl]

                                            new_b_00 = cell_b_00 * b_00 + cell_b_10 * b_01
                                            new_b_01 = cell_b_01 * b_00 + cell_b_11 * b_01
                                            new_b_10 = cell_b_00 * b_10 + cell_b_10 * b_11
                                            new_b_11 = cell_b_01 * b_10 + cell_b_11 * b_11

                                            if new_b_00 > epsi and new_b_11 > epsi:

                                                angle_b = pi/2 - atan(new_b_01 / new_b_00) - atan(new_b_10 / new_b_11)

                                                if abs(angle_a - angle_b) < toll_rotation:

                                                    lenght_a_0 = sqrt(new_a_00**2 + new_a_01**2)
                                                    lenght_a_1 = sqrt(new_a_10**2 + new_a_11**2)
                                                    lenght_b_0 = sqrt(new_b_00**2 + new_b_01**2)
                                                    lenght_b_1 = sqrt(new_b_10**2 + new_b_11**2)

                                                    for i in range(2):

                                                        if i == 0:
                                                            diff_0 = abs(lenght_a_0 - lenght_b_0) / lenght_a_0
                                                            diff_1 = abs(lenght_a_1 - lenght_b_1) / lenght_a_1
                                                            invert = False
                                                        elif 1 == 1:
                                                            diff_0 = abs(lenght_a_0 - lenght_b_1) / lenght_a_0
                                                            diff_1 = abs(lenght_a_1 - lenght_b_0) / lenght_a_1
                                                            invert = True

                                                        if diff_0 < stretch_max and diff_1 < stretch_max:

                                                            deform_nrg = diff_0**2 + diff_1**2

                                                            j = 0
                                                            for i in range(len(areas)):
                                                                if area < areas[i] - 1e-2 or deform_nrg < deform_nrgs[i] - 1e-6:
                                                                    j += 1

                                                            if j == len(areas):
                                                                vect_a = [[a_00, a_01], [a_10, a_11]]
                                                                vect_b = [[b_00, b_01], [b_10, b_11]]
                                                                print('STRUCTURE', len(areas)+1)
                                                                print('surface vectors a =', vect_a)
                                                                print('surface vectors b =', vect_b)
                                                                print('invert axis b  =', invert)
                                                                print("deformation x' = {0:6.3f} %".format(diff_0 * 100))
                                                                print("deformation y' = {0:6.3f} %".format(diff_1 * 100))
                                                                print('slab area = {0:7.4f}\n'.format(area))
                                                                areas.append(area)
                                                                deform_nrgs.append(deform_nrg)

                                                                if deform_nrg == min(deform_nrgs):
                                                                    vect_a_opt = vect_a
                                                                    vect_b_opt = vect_b
                                                                    invert_opt = invert
                                                                    match_dimensions = True

    return vect_a_opt, vect_b_opt, invert_opt, match_dimensions

################################################################################
# FIND INVERSION CENTRE CYTHON
################################################################################

@cython.cdivision(True)    # use C semantics for division
@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
def find_inversion_centre_cython(double [:, ::1] positions, list symbols):

    print('USING CYTHON\n')

    cdef:
        unsigned int i, j
        double x, y, z, a_x, a_y, a_z, b_x, b_y, b_z
        int n_atoms = positions.shape[0]
        int n_centres
        int [::1] indices
        list c_matrix = []

    for i in range(n_atoms):
        for j in range(i, n_atoms):
            if symbols[i] == symbols[j]:
                x = (positions[i][0] + positions[j][0]) / 2.
                y = (positions[i][1] + positions[j][1]) / 2.
                z = (positions[i][2] + positions[j][2]) / 2.
                c_matrix.append((x, y, z))

    n_centres = len(c_matrix)
    print('number of centres =', n_centres)

    indices = np.zeros(n_centres, dtype = np.int32)

    for i in range(n_centres):
        for j in range(i, n_centres):
            a_x = round(c_matrix[i][0] * 1e3)
            a_y = round(c_matrix[i][1] * 1e3)
            a_z = round(c_matrix[i][2] * 1e3)
            b_x = round(c_matrix[j][0] * 1e3)
            b_y = round(c_matrix[j][1] * 1e3)
            b_z = round(c_matrix[j][2] * 1e3)
            if a_x == b_x and a_y == b_y and a_z == b_z:
                indices[i] += 1

    print('number of occurrences =', max(indices))

    centre = c_matrix[np.argmax(indices)]

    return centre

################################################################################
# END
################################################################################
