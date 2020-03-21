import sys

import numpy as np
from copy import copy

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor, as_completed

from helper_functions import *

def surface_area_calculation(l_axis, d_axis, bins, bin_range, p_init, i):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax_hist = fig.add_subplot(122)

    if i % 60 == 0:
        print("angle: {}".format(i)) 

    h_theta = rotate_vector_by_degree(l_axis, d_axis, i)
    w_theta = np.cross(l_axis, h_theta)

    # 3D to 2D
    # h_(theta)l-plane 
    p_transformed = transform_axis_pointcloud(p_init, h_theta, l_axis, w_theta)
    H = ax_hist.hist2d(p_transformed[:, 0], p_transformed[:, 1], bins=bins, range=bin_range)
    fig.colorbar(H[3], ax=ax_hist)
    plt.close()
    return (np.count_nonzero(H[0]), i)

class LeafAxisDetermination(object):
    def __init__(self, pointcloud):
        self.p_init = copy(pointcloud)
        self.surface_area_list = []

    def pca_pointcloud(self, pointcloud):
        """
        # 1. Apply PCA (component_1 -> l-axis )
        # 2. Calculate the angle between the centroid of p_init and determine the l-axis

        """
        assert(pointcloud.shape[1] == 3)
        centroid = np.mean(pointcloud, axis=0)

        # Translate the pointcloud to its centroid
        pointcloud_0 = pointcloud - centroid

        pca = PCA(n_components=3)
        pca.fit(pointcloud_0)

        # PCA does not gurantee the right hand coordinate vector direction
        component_1, component_2, component_3 = pca.components_

        # so we calclate the the 2nd pc by cross calculation of 1st pc and 3rd pc
        # this calculation is not neccesary for leaf flattening, but we calculate it to check the 
        # calculation is correct by visualizing the vectors.
        component_2 = np.cross(component_1, component_3) * -1.0

        # assure the component_1 vector (l-axis) to point the centroid
        check_component_1_radian = calculate_radian(centroid[0:2], component_1[0:2])
        if check_component_1_radian >= np.pi / 2.0:
            component_1 = -1.0 * component_1
            component_3 = -1.0 * component_3 
            print("Component 1 has been reversed.") 
        
        check_component_3_radian = calculate_radian(np.array([0.0, 0.0, 1.0]), component_3)
        if check_component_3_radian >= np.pi / 2.0:
            component_2 = -1.0 * component_2
            component_3 = -1.0 * component_3
            print("Component 3 has been reversed.")
        
        return component_1, component_2, component_3
        
    
    def process(self):
        """
        Leaf Axis Determination Process

        # 1. Apply PCA (component_1 -> l-axis ).
        # 2. Calculate the angle between the centroid of p_init and determine the l-axis.
        # 3. Shift the origin of p_init.
        # 4. Generate initial axis d by calculating the axis perpendicular to l-axis.
        # 5. Rotate d-axis around the l-axis for angle theta, to create h_(theta)-axis.
        # 6. Compute the surface area of point cloud projected to h_(theta)l-plane.
        # 7. Determine the z-axis element of direction vector h.

        """ 

        # 1 and 2
        l_axis, component_2, component_3 = self.pca_pointcloud(self.p_init)

        # 3
        self.p_init = self.p_init - np.mean(self.p_init, axis=0)

        # 4
        # we use the component_3 we calculated above as the d_axis
        d_axis = component_3

        # we initially transform the point cloud to ld-plane to 
        # determine the rectangular grid
        candidate_w_axis = np.cross(l_axis, d_axis)
        p_trans_0 = transform_axis_pointcloud(self.p_init, candidate_w_axis, l_axis, d_axis)

        xmin = np.min(p_trans_0[:, 0]) # the l-axis
        xmax = np.max(p_trans_0[:, 0]) # the l-axis

        ymin = np.min(p_trans_0[:, 1]) # the d_axis
        ymax = np.max(p_trans_0[:, 1]) # the d_axis

        bins = np.array([100, 100])
        bin_range = np.array([[xmin, xmax], [ymin, ymax]])

        # 5 and 6
        with ProcessPoolExecutor(max_workers=4) as executor:
            res = [executor.submit(surface_area_calculation, l_axis, d_axis, bins, bin_range, self.p_init, i)
                    for i in range(180)]
        
            for future in as_completed(res):
                self.surface_area_list.append(future.result()[0])
        
        # 7
        surface_area = np.array(self.surface_area_list)
        smallest_area_angle = np.argmin(surface_area)

        h_axis = rotate_vector_by_degree(l_axis, d_axis, smallest_area_angle)

        check_h_axis_radian = calculate_radian(h_axis, np.array([0.0, 0.0, 1.0]))
        if check_h_axis_radian >= np.pi / 2.0:
            print("h axis is not pointing leaf top. Reverse it.")
            h_axis = h_axis * -1.0

        w_axis = np.cross(l_axis, h_axis) 

        return w_axis, l_axis, h_axis