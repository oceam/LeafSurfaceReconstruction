import os

import numpy as np
from scipy import optimize
from helper_functions import *

def quadratic(parameters, x, y): 
    """
    fit quadratic function
    """
    a = parameters[0] 
    b = parameters[1]
    c = parameters[2]

    func = x*x*a + x*b + c
    residual = y - func 
    return residual

def cubic(parameters, x, y):
    """ 
    fit cubic equation
    """
    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]

    func = x*x*x*a + x*x*b + x*c + d
    residual = y - func 
    return residual


class LeafFlattening(object):
    def __init__(self, pointcloud, output_folder):
        self.pointcloud = pointcloud
        self.output_folder = output_folder

    def bending_skeleton_extraction(self, x, y, sample_num=50):
        """
        fit the skeleton function to the 2D point cloud 
        to the bending direction
        using the cubic funtion as a skeleton function
        """
        init_parameters = [0.0, 0.0, 0.0]
        result = optimize.leastsq(quadratic, init_parameters, args=(x, y))
        parameters = result[0]

        # sample skeleton points
        min_x = np.min(x)
        max_x = np.max(x) 

        # to capture the leaf tips that curls to the rolling directon
        # we sample skeleton points from a bit longer section
        sub_length = (max_x - min_x) * 0.05 
        skeleton_x = np.linspace(min_x, max_x + sub_length, sample_num)
        skeleton_y = []

        for s_x in skeleton_x:
            skeleton_y.append(s_x*s_x*parameters[0] + s_x*parameters[1] + parameters[2])

        skeleton_y = np.asarray(skeleton_y)

        return np.concatenate([skeleton_x[:, np.newaxis], skeleton_y[:, np.newaxis]], axis=1), parameters

    def rolling_skeleton_extraction(self, x, y, sample_num=50):
        """
        fit the skeleton function to the 2D point cloud
        to the rolling direction
        using the quadratic function as a skeleton function
        """
        init_parameters = [0.0, 0.0, 0.0, 0.0]
        result = optimize.leastsq(cubic, init_parameters, args=(x, y))
        parameters = result[0]

        # sample skeleton points
        min_x = np.min(x) 
        max_x = np.max(x)

        skeleton_x = np.linspace(min_x, max_x, 50)
        skeleton_y = []

        for s_x in skeleton_x:
            skeleton_y.append(s_x*s_x*s_x*parameters[0] + s_x*s_x*parameters[1] + s_x*parameters[2] + parameters[3])
        
        skeleton_y = np.asarray(skeleton_y)

        return np.concatenate([skeleton_x[:, np.newaxis], skeleton_y[:, np.newaxis]], axis=1), parameters

    def calculate_one_strip(self, pivot, pivot_prime, pc_2d):
        pc_tmp = pc_2d - pivot

        pivot_vector = pivot_prime - pivot
        pivot_vector /= np.linalg.norm(pivot_vector)

        theta = calculate_angle_between_vectors(pivot_vector, np.array([1.0, 0.0]))
        rotation_vector = calc_rotation_matrix_2d(theta)
        rotated_points = np.dot(rotation_vector, pc_tmp.T).T

        rotated_pivot_prime = np.dot(rotation_vector, pivot_prime - pivot)

        strip_points_ind = np.where((rotated_points[:, 0] >= 0) & (rotated_points[:, 0] <= rotated_pivot_prime[0])) 
        return rotated_points, strip_points_ind, rotated_pivot_prime, rotation_vector

    def process(self):
        """
        perfrom leaf flattening
        """
        # we first use the lh-plane to extract and flatten the leaf 
        # in the bending direction
        pc_lh = self.pointcloud[:, 1:3]

        # Exract skeleton points
        # lh-plane
        skeleton_points, _ = self.bending_skeleton_extraction(pc_lh[:, 0], pc_lh[:, 1])

        # Extract alignment skeleton points
        # the lw-plane
        alignment_skeleton_points_2d, _ = self.bending_skeleton_extraction(self.pointcloud[:, 1], self.pointcloud[:, 0])

        # we concatenate the w-axis element of alignment_skeleton_points to skeleton_points_bend
        # to generate the alignment_skeleton_points
        alignment_skeleton_points_3d = np.concatenate([alignment_skeleton_points_2d[:, 1][:, np.newaxis], skeleton_points], axis=1)

        """ 
        # remove this comment to chech the fitted skeleton points and the leaf point cloud
        # visualize_3D_pointcloud_2D(pc_lh[:, 0], pc_lh[:, 1], skeleton_points, "skeleton_points",
        #                             save_dir=self.output_folder)
        """
        skeleton_length_bend = 0 # t_k for bending direction flattening

        one_way_flattened_pc = np.empty((0, 3))
        two_way_flattened_pc = np.empty((0, 3))

        one_way_leaf_area = 0
        two_way_leaf_area = 0

        for i in range(len(skeleton_points) - 1):
            pivot = skeleton_points[i] # s_i
            pivot_prime = skeleton_points[i + 1] # s_(i+1) 
            rotated_points, strip_points_ind, rotated_pivot_prime, rotation_vector = self.calculate_one_strip(pivot, pivot_prime, pc_lh)
            strip_points_lh = rotated_points[strip_points_ind]

            # we call the w-axis element from the point cloud
            strip_points_w = self.pointcloud[strip_points_ind][:, 0]
            strip_points = np.concatenate([strip_points_w[:, np.newaxis], strip_points_lh], axis=1)
            strip_points[:, 1] = strip_points[:, 1] + skeleton_length_bend

            if (strip_points.shape[0] != 0):
                one_way_flattened_pc = np.concatenate([one_way_flattened_pc, strip_points], axis=0)
            
            """ remove this comment to check the process of rotating the lef point cloud
            visualize_3D_pointcloud_2D(rotated_points[:, 0], rotated_points[:, 1],
                                        np.dot(rotation_vector, (skeleton_points - pivot).T).T, "check")
            """ 

            s_wl = alignment_skeleton_points_3d[i]

            # in the rolling direction
            if (strip_points.shape[0] > 5):
                strip_pc_wh = np.concatenate([strip_points[:, 0][:, np.newaxis], strip_points[:, 2][:, np.newaxis]], axis=1)

                # use the wh-plane to extract and flatten the leaf
                strip_skeleton_points, param_r = self.rolling_skeleton_extraction(strip_pc_wh[:, 0], strip_pc_wh[:, 1])

                flattened_strip = np.empty((0, 3))
                flattened_strip_skeleton_points = np.zeros((1, 2))
                skeleton_length_roll = 0 # t_k for rolling direction flattening

                # we search for landmark point
                dist_list = []
                for j in range(len(strip_skeleton_points)):
                    dist_list.append(np.linalg.norm(strip_skeleton_points[j] - np.array(s_wl[0], s_wl[2])))
                landmark_index = np.argmin(np.array(dist_list))

                # flatten to the rolling direction
                for j in range(len(strip_skeleton_points) - 1):
                    strip_rotated_points, strip_rotated_points_ind, strip_rotated_pivot_prime, _ = self.calculate_one_strip(strip_skeleton_points[j], strip_skeleton_points[j + 1], strip_pc_wh)
                    flattened_strip_points_w = strip_rotated_points[strip_rotated_points_ind][:, 0]
                    flattened_strip_points_l = strip_points[strip_rotated_points_ind][:, 1]
                    flattened_strip_points_h = strip_rotated_points[strip_rotated_points_ind][:, 1]

                    flattened_strip_points = np.concatenate([flattened_strip_points_w[:, np.newaxis], 
                                                            flattened_strip_points_l[:, np.newaxis],
                                                            flattened_strip_points_h[:, np.newaxis]], axis=1)
                    flattened_strip_points[:, 0] = flattened_strip_points[:, 0] + skeleton_length_roll

                    # we save the length to the landmark point
                    if (j == landmark_index):
                        strip_center = skeleton_length_roll
                    
                    flattened_strip = np.concatenate([flattened_strip, flattened_strip_points])
                    flattened_strip_skeleton_points = np.concatenate((flattened_strip_skeleton_points, 
                                                            strip_rotated_pivot_prime[np.newaxis] + np.array([skeleton_length_roll, 0.0])), axis=0)
                    skeleton_length_roll += np.linalg.norm(strip_rotated_pivot_prime)               

                if (landmark_index == len(strip_skeleton_points) - 1):
                    strip_center = skeleton_length_roll

                # we translate the origin of flattened_strip to the position of the landmark point
                if (flattened_strip.shape[0] >= 2):
                    flattened_strip[:, 0] = flattened_strip[:, 0] - strip_center
                
                two_way_flattened_pc = np.concatenate([two_way_flattened_pc, flattened_strip])
            
            else:
                flattened_strip = np.empty((0, 3))

            skeleton_length_bend += np.linalg.norm(rotated_pivot_prime)

            ############## Calcuclate Leaf area ##############

            # area for one_way_flattened_pc
            if (strip_points.shape[0] >= 2):
                leaf_height = np.linalg.norm(pivot_prime - pivot)
                leaf_width = np.max(strip_points[:, 0]) - np.min(strip_points[:, 0])
                one_way_leaf_area += leaf_height * leaf_width

                if (flattened_strip[:, 0].shape[0] >= 2):
                    two_way_leaf_width = np.max(flattened_strip[:, 0]) - np.min(flattened_strip[:, 0])

                    # we use the same leaf_height value
                    two_way_leaf_area += leaf_height * two_way_leaf_width
            
            else:
                one_way_leaf_area += 0
                two_way_leaf_area += 0
        
        print(one_way_leaf_area)
        print(two_way_leaf_area)

        return one_way_flattened_pc, two_way_flattened_pc