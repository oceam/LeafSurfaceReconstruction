import os
import numpy as np
import matplotlib.pyplot as plt

def calc_rotation_matrix_2d(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    a11 = cos_theta
    a12 = -sin_theta
    a21 = sin_theta
    a22 = cos_theta

    return np.array([[a11, a12], [a21, a22]])

def calculate_angle_between_vectors(a, b):
    # Ref
    outer_cross = np.cross(a, b)
    dot_product = np.dot(a, b)
    theta = np.arctan2(outer_cross, dot_product)
    return theta

def calc_rotation_matrix(axis, theta):
        """
        axis: norm 1 rotation of axis vector
        theta: 0 to 2*pie rotation radian
        """ 
        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)
        n1, n2, n3 = axis

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        a11 = cos_theta + n1*n1*(1 - cos_theta)
        a12 = n1*n2*(1 - cos_theta) - n3*sin_theta
        a13 = n1*n3*(1 - cos_theta) + n2*sin_theta 
        a21 = n2*n1 * (1 - cos_theta) + n3*sin_theta
        a22 = cos_theta + n2*n2 * (1 - cos_theta)
        a23 = n2*n3*(1 - cos_theta) - n1*sin_theta
        a31 = n3*n1*(1 - cos_theta) - n2*sin_theta
        a32 = n3*n2*(1 - cos_theta) + n1*sin_theta
        a33 = cos_theta + n3*n3*(1 - cos_theta)

        return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

def rotate_vector_by_degree(axis, vector, degree):
    # this only works on 3D vector
    theta = np.deg2rad(degree)
    rot_mat = calc_rotation_matrix(axis, theta)
    translated_vector = np.matmul(rot_mat, vector)
    return translated_vector

def transform_axis_pointcloud(pointcloud, x_axis, y_axis, z_axis):

    mat = np.array([[x_axis[0], y_axis[0], z_axis[0]],
                    [x_axis[1], y_axis[1], z_axis[1]],
                    [x_axis[2], y_axis[2], z_axis[2]]])

    mat_inv = np.linalg.pinv(mat)
    pointcloud_transformed = np.matmul(mat_inv, pointcloud.T).T
    return pointcloud_transformed

def calculate_radian(vector1, vector2):
    dot = np.dot(vector1, vector2)
    norm = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos = dot / norm
    return np.arccos(np.clip(cos, -1.0, 1.0))

def visualize_3D_pointcloud_2D(x, y, skeleton_points, figname, save_dir=None, xlim=None, ylim=None):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="black", s=1)
    if skeleton_points is not None:
        ax.scatter(skeleton_points[:, 0], skeleton_points[:, 1], color="blue", s=15)

    ax.set_aspect("equal", "datalim")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.title(figname)
    if save_dir:
        img_name = figname + ".png"
        plt.savefig(os.path.join(save_dir, img_name))
    else:
        plt.show()
    plt.close()
