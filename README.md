# End2End Gaussian Policy
This repo consists of two part.
1. 3DGS with semantic features from multiple caibrated RGBD cameras
2. End2End End-Effector pose estimation from a perceiver-based policy
## Construct the feature-geo representation from multiple RGBD views
Here we use a geo-reconstruction method similar to 3DGS. Our approach consists with three stages
1. Initial an aligned point cloud from Multi-view depth-image and its corresponding pixel-level feature map
2. Run 3dgs to fine-tune the feature point cloud with rgb and depth rendering loss
3. Use the constructed feature field as policy input and build an end-to-end policy
## Data format
In this part, we discuss the details of training data. Our original training data consists with three parts
1. depth/rgb images are stored in images/depths dir
2. calibrated camera info(6 dimension vector includes position+rotatiobn)
3. keyframe end-effector pose(optional, when do validation this is not required)
## prerequisite to run the code
Before run the whole pipeline, you need to clone the repo recursive and compile the submodule/diff_gaussian_rasterization and submodule/simple_knn in a machine with CUDA support.
