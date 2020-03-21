import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from leaf_flattening import LeafFlattening
from leaf_axis_determination import LeafAxisDetermination
from helper_functions import *

def check_pca_axis(x_axis, y_axis, z_axis, centroid):
    points = []
    for i in range(500):
        x = centroid + x_axis * 0.1 * i
        y = centroid + y_axis * 0.1 * i
        z = centroid + z_axis * 0.1 * i
        points.append(np.hstack( (x, np.array([255, 0, 0]) ) ) ) # R w
        points.append(np.hstack( (y, np.array([0, 255, 0]) ) ) ) # G l
        points.append(np.hstack( (z, np.array([0, 0, 255]) ) ) ) # B h
    
    return np.asarray(points)

class LeafProcess(object):
    def __init__(self, path_params):
        self.leaf_path = path_params["leaf_path"]
        self.leaf_name = path_params["leaf_name"]
        self.output_folder = path_params["output_folder"]

        if os.path.exists(self.output_folder) != True:
            os.makedirs(self.output_folder)
        
        self.output_name = path_params["output_name"]

        self.one_way_leaf_area = {}
        self.two_way_leaf_area = {}

        # load files
        df = pd.read_csv(self.leaf_path, names=("x", "y", "z"))
        self.pointcloud = df[["x", "y", "z"]].values

    def reconstruction(self): 
        # x_axis, y_axis, z_axis = calculate_plane_from_image(self.pointcloud)
        leafAxisDetermination = LeafAxisDetermination(self.pointcloud)
        w_axis, l_axis, h_axis = leafAxisDetermination.process()

        # translate point cloud to its centroid
        self.p = self.pointcloud - np.mean(self.pointcloud, axis=0)
        
        # translate to leaf coordinate
        self.p = transform_axis_pointcloud(self.p, w_axis, l_axis, h_axis)

        # leaf flattening
        leafFlattening = LeafFlattening(self.p, self.output_folder)
        one_way_flattened_pc, two_way_flattened_pc = leafFlattening.process()

        one_way_img_name = self.leaf_name + "_one_way"
        two_way_img_name = self.leaf_name + "_two_way"

        # capture the leaf surface by projecting the pointcloud to the wl-plane
        visualize_3D_pointcloud_2D(one_way_flattened_pc[:, 0], one_way_flattened_pc[:, 1], None, 
                                    one_way_img_name, save_dir=self.output_folder, ylim=None)
        visualize_3D_pointcloud_2D(two_way_flattened_pc[:, 0], two_way_flattened_pc[:, 1], None, 
                                    two_way_img_name, save_dir=self.output_folder, ylim=None)
        
        return one_way_flattened_pc, two_way_flattened_pc