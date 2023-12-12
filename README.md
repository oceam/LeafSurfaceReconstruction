# A Robust Leaf Surface Reconstruction Method for 3D Plant Phenotyping
This is a support page of paper submmited to "Plant Phenomics" as an Original Research.

> ANDO,R., OZASA,Y., GUO, W.(2021). Robust Surface Reconstruction of Plant Leaves from 3D Point Clouds. Plant Phenomics. 2021;2021:DOI:10.34133/2021/3184185

---

# Test Environment
The test environment for this repository is 

- Python 3.6

# Quick Start
To reconstruct the leaf surface from the point cloud data located at data/soybean is 

`python reconstruction.py --input data/soybean --output output/soybean
`

This will create an output/soybean directory and produces the result for each leaf
that exist in the data/soybean directory.

# Outputs

The source code will produce "one_way_surface.txt" and "two_way_surface.txt", which is the leaf surface flattened to the bending direction and the leaf surface flattened to both the bending and the rolling direction respectively.

The source code will also produce two images, which are the leaf surfaces projected onto the wl-plane for each results. 
