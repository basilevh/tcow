'''
Tools / utilities / helper methods pertaining to camera projections and other 3D stuff.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utils/'))

from __init__ import *

# Library imports.
import numpy as np


def box_to_tf_matrix(box, rows):
    '''
    :param box (8, 3) array: All corners in XYZ space of 3D cube surrounding object.
    :param (tf_matrix, rows).
        tf_matrix (4, 4) array: Coordinate transformation matrix from object space to world space.
        rows (3) array: Indices of rows in box that form an edge with the first row (= origin).
    '''
    # We make minimal assumptions about the ordering of the 3D points, except that the first two
    # rows must make up an edge of the box. Then, we look for the next two orthogonal vectors.
    origin = box[0]

    if rows is None:
        axis1 = box[1] - origin
        axis2 = None
        axis3 = None
        row1 = 1
        row2 = None
        row3 = None

        for i in range(2, 8):
            cand_axis = box[i] - origin
            if axis2 is None:
                if np.abs(np.dot(axis1, cand_axis)) < 1e-7:
                    axis2 = cand_axis
                    row2 = i
            elif axis3 is None:
                if np.abs(np.dot(axis1, cand_axis)) < 1e-7 and np.abs(np.dot(axis2, cand_axis)) < 1e-7:
                    axis3 = cand_axis
                    row3 = i

        assert axis2 is not None and axis3 is not None, \
            'Could not find orthogonal axes for object_box'
        rows = np.array([row1, row2, row3])

    else:
        axis1 = box[rows[0]] - origin
        axis2 = box[rows[1]] - origin
        axis3 = box[rows[2]] - origin

    object_to_world = np.stack([axis1, axis2, axis3, origin], axis=1)
    object_to_world = np.concatenate([object_to_world, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    # Sanity check while debugging: origin + axis1 must be close to object_to_world @ [1, 0, 0, 1].
    # NOTE: object_to_world is generally not orthonormal, because the axis lengths follow the size
    # of the container box, not unit vectors.

    return (object_to_world, rows)


def get_containment_fraction_approx(inside_box, outside_box):
    '''
    Calculates a sampled approximation of how much volume of a non-aligned 3D bounding box of a
        candidate object intersects (i.e. is inside of) that of a reference object.
    :param inside_box (8, 3) array of float: All corners in XYZ space of candidate containee cube.
    :param outside_box (8, 3) array of float: All corners in XYZ space of reference container cube.
    :return cflb (float).
    '''
    # NEW: Work with sampling. This is kind of brute-force, but at least it is simple and correct.
    # https://stackoverflow.com/questions/1827489/numpy-meshgrid-in-3d
    (x, y, z) = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6), np.linspace(0, 1, 6),
                            indexing='ij')
    xyz = np.stack([x, y, z], axis=-1).reshape((-1, 3))  # (216, 3).
    xyz_homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)  # (216, 4).

    # Study the inside box in the coordinate system of the outside box.
    (outside_to_world, rows) = box_to_tf_matrix(outside_box, None)
    (inside_to_world, rows) = box_to_tf_matrix(inside_box, None)
    world_to_outside = np.linalg.inv(outside_to_world)
    inside_to_outside = np.matmul(world_to_outside, inside_to_world)
    # # NOTE: Unlike outside_to_world, world_to_outside (and inside_to_outside) are generally not
    # even orthogonal, because outside_to_world is not orthonormal!

    xyz_warped = np.matmul(inside_to_outside, xyz_homo.T).T
    assert np.all(np.abs(xyz_warped[..., -1] - 1.0) < 1e-5), \
        'Homogeneous coordinate is not 1'
    xyz_warped = xyz_warped[..., 0:3]
    points_contained = np.logical_and(np.all(xyz_warped >= 0.0, axis=1),
                                      np.all(xyz_warped <= 1.0, axis=1))
    cf_approx = np.mean(points_contained.astype(np.float32))

    return cf_approx
