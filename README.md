# 3D-SemiDeformable-ObjectTracking

This repository contains a suite of programs meant to aid in the detection of semi-deformable objects. The object used in this repository consisted of a bell pepper. It was used to develop my Master's Thesis. 

<div class="sketchfab-embed-wrapper">
  <iframe title="Bp4" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/0d6db7c6317a4b7fbb59a03e61d79c5c/embed"> 
  </iframe> 
  <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;"> 
    <a href="https://sketchfab.com/3d-models/bp4-0d6db7c6317a4b7fbb59a03e61d79c5c?utm_medium=embed&utm_campaign=share-popup&utm_content=0d6db7c6317a4b7fbb59a03e61d79c5c" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> Bp4 </a> 
    by <a href="https://sketchfab.com/gustavodlra1999?utm_medium=embed&utm_campaign=share-popup&utm_content=0d6db7c6317a4b7fbb59a03e61d79c5c" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> gustavodlra </a> 
    on <a href="https://sketchfab.com?utm_medium=embed&utm_campaign=share-popup&utm_content=0d6db7c6317a4b7fbb59a03e61d79c5c" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;">Sketchfab</a>
  </p>
</div>


## Python Requirements

The code in general was developed in a Windows 11 machine. It was developed using the Anaconda python distribution. An Anaconda virtual environment with Python 3.10.13 was used. A requirements.txt file is included in order to allow for a recreation of the environment used for the development of this code. 

## The Reference Notebooks Folder

The Reference Notebooks Folder contains 6 Jupyter notebooks that detail the process that allowed to perform the detection and pose estimation of a bell pepper in 3-D. The notebooks should be read in the following order: 

1. **img_segmentation_pc_creation.ipynb:** This notebook illustrates the process necessary for the extraction of color and depth data from images captured by an Azure Kinect. The result of this notebook is a colored point cloud of the object of interest.
2. **create_pc_from_mesh.ipynb:** The workflow in this notebook takes a 3D model in STL format as input. The model is then transformed into a point cloud with the a number of points specified in the notebook and saved in a specified directory.
3. **pc_aligning_and_scaling:** This notebook performs the key part of this process. It illustrates and explains the series of steps necessary to accurately deform the point cloud of a canonical 3D model with recognizable characteristics of a deformable object. A centroid-to-furthest-point defining feature line is found in both the point cloud of the canonical model and in the cloud of the object scanned by the Kinect. These lines are then used in order to perform an orientation alignment, which is later refined. Once both clouds are properly aligned, the point cloud of the canonical model is then scaled to match the dimensions of the Kinect-scanned real-world object. The output of this notebook is an accurately scaled canonical model. 
4. **pose_registration.ipynb:** After having obtained the scaled and aligned model point cloud in the prior notebook, this notebook uses the RAndom SAmple Consensus (RANSAC) and Iterative Closest Points (ICP) algorithms to register and approximate the pose of the real world point cloud by aligning the scaled canonical model to it using these algorithms in the sequence they were mentioned. RANSAC performs a quality initial pose estimation which is then refined via the use of the ICP algorithm. This notebook outputs a transformation matrix that can be used to align the model point cloud with the best pose that this notebook could obtain.
5. **generate2D_rep.ipynb:** The scaled and deformed point cloud is put into the pose given by the prior notebook. The point cloud is then transformed using an ortographic projection with matplotlib in order to generate a 2D representation that can be superimposed onto the color image.
6. **overlay_2d_og_img.ipynb:** This code uses the orthogonal view generated in the prior notebook to superimpose it in the centroid of the object in the color image. Allowing for an in-context visualization.  
