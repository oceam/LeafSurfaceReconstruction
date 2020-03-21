# Source Code for Leaf Flattening Method

import os
import sys
import csv
import glob
import time
import datetime
import argparse

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "utils"))

from leaf_process import LeafProcess

print(BASE_DIR)

parser = argparse.ArgumentParser(description="Leaf Flattening")
parser.add_argument("--input", default="data\\sugarbeet")
parser.add_argument("--output", default="output\\sugarbeet")
args = parser.parse_args()

# timestr = time.strftime("%Y%m%d-%H%M%S")
# timestr = "tmp"
output_folder_name = args.output
print(output_folder_name)

INPUT_DATA_DIR = os.path.abspath(args.input)
OUTPUT_DIR = os.path.join(BASE_DIR, output_folder_name)

# create directory if output directory does not exists
if os.path.exists(OUTPUT_DIR) != True:
    os.makedirs(OUTPUT_DIR)

def reconstruct(path_params):
    """
    function to flatten and reconstruct the surface of leaf point cloud
    """
    leafProcess = LeafProcess(path_params)

    # one_way -> the leaf surface for flattening the leaf to the bending direction
    # two_way -> the leaf surface for flattening the leaf to both the bending and rolling direction
    one_way, two_way = leafProcess.reconstruction()

    return one_way, two_way

if __name__ == "__main__":
    leafs_path = sorted(os.listdir(INPUT_DATA_DIR))
    # print(leafs_path)

    with open(os.path.join(OUTPUT_DIR, "result.csv"), "w") as f:
        writer = csv.writer(f, lineterminator="\n")

        start = time.perf_counter()
        for leaf_path in leafs_path:
            # create path parameter 
            path_params = {}
            path_params["leaf_path"] = os.path.join(INPUT_DATA_DIR, leaf_path)
            path_params["leaf_name"] = leaf_path.split(".")[0]
            path_params["output_name"] = path_params["leaf_name"] + "_fig" 
            path_params["output_folder"] = os.path.join(OUTPUT_DIR, path_params["leaf_name"])
    
            if os.path.exists(path_params["output_folder"]) != True:
                os.makedirs(path_params["output_folder"])
            
            else:
                print("skipped: {}".format(path_params["output_folder"]))
            
            result1, result2 = reconstruct(path_params)

            print( "result 1: {}".format(result1.shape) )
            print( "result 2: {}".format(result2.shape) )
            
            # save the leaf surface
            np.savetxt(os.path.join(path_params["output_folder"], "one_way_surface.txt"), result1,
                        delimiter=",")
            np.savetxt(os.path.join(path_params["output_folder"], "two_way_surface.txt"), result2,
                        delimiter=",")

            # writer.writerow(result)
            print( "{}: Done".format(leaf_path) )

    elapsed_time = time.perf_counter() - start 
    print("Finished: {0}".format(elapsed_time) + "[sec]")
        