# Originally named ransac_integration.py

# 1. Import the libraries
import cv2 
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
import numpy as np
import matplotlib.pyplot as plt
import threading
from collections import deque
import json
import open3d as o3d
import time
import os
import sys
import copy

# 3D file to import: 
model_path = "output_10000.ply"
source = o3d.io.read_point_cloud(model_path)

# Filtering values
MIN_DEPTH = 0
MAX_DEPTH = 1000

lower_green = np.array([18, 7, 30])
upper_green = np.array([94, 255, 148])




# 2. Image processing functions
def filter_color(image, lower_bound, upper_bound):
    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Create a mask using the given bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # Apply the mask
    #result = cv2.bitwise_and(image, image, mask=mask)
    return mask

def filter_depth(image, min_distance, max_distance):
    # Create a mask using the given bounds
    mask = ((image > min_distance) & (image < max_distance)).astype(np.uint8) * 255
    return mask

# Trackbar callback functions
def update_lower_h(val):
    global lower_green
    lower_green[0] = val

def update_lower_s(val):
    global lower_green
    lower_green[1] = val

def update_lower_v(val):
    global lower_green
    lower_green[2] = val

def update_upper_h(val):
    global upper_green
    upper_green[0] = val

def update_upper_s(val):
    global upper_green
    upper_green[1] = val

def update_upper_v(val):
    global upper_green
    upper_green[2] = val

# Create a window for the trackbars
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

# Create the trackbars
cv2.createTrackbar("Lower H", "Trackbars", lower_green[0], 179, update_lower_h)
cv2.createTrackbar("Lower S", "Trackbars", lower_green[1], 255, update_lower_s)
cv2.createTrackbar("Lower V", "Trackbars", lower_green[2], 255, update_lower_v)
cv2.createTrackbar("Upper H", "Trackbars", upper_green[0], 179, update_upper_h)
cv2.createTrackbar("Upper S", "Trackbars", upper_green[1], 255, update_upper_s)
cv2.createTrackbar("Upper V", "Trackbars", upper_green[2], 255, update_upper_v)



# 3. Real time Point Cloud visualization
# Point cloud visualization in real time using numpy
def get_point_cloud_vectorized(depth_image, intrinsic_matrix):
    height, width = depth_image.shape
    
    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to camera coordinates
    x_cam = (x - intrinsic_matrix[0, 2]) * depth_image / intrinsic_matrix[0, 0]
    y_cam = (y - intrinsic_matrix[1, 2]) * depth_image / intrinsic_matrix[1, 1]
    
    # Stack the coordinates together
    points = np.stack((x_cam, y_cam, depth_image), axis=-1)

    # Apply depth range threshold and remove points with zero depth
    mask = (depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH) & (depth_image != 0)
    return points[mask]

transformation_matrix = [[1, 0, 0, 0], 
                         [0, -1, 0, 0], 
                         [0, 0, -1, 0], 
                         [0, 0, 0, 1]]

# 4. Centroid to defining feature. 
def compute_centroid(point_cloud): 
    return np.mean(point_cloud.points, axis=0)

def identify_peduncle_point(pcd): 
    z_coordinates = np.asarray(pcd.points)[:,2]

    threshold = np.percentile(z_coordinates, 98)
    top_points = np.asarray(pcd.points)[z_coordinates > threshold]

    return np.mean(top_points, axis=0)

def identify_peduncle_point_scan(pcd): 
    y_coordinates = np.asarray(pcd.points)[:,0]

    threshold = np.percentile(y_coordinates, 98)
    top_points = np.asarray(pcd.points)[y_coordinates > threshold]

    return np.mean(top_points, axis=0)

def compute_line(pcd):
    centroid = compute_centroid(pcd)
    peduncle_point = identify_peduncle_point(pcd)

    direction_vector = peduncle_point - centroid
    normalized_vector = direction_vector / np.linalg.norm(direction_vector)
    return normalized_vector, centroid, peduncle_point

def compute_line_scan(pcd):
    centroid = compute_centroid(pcd)
    peduncle_point = identify_peduncle_point_scan(pcd)

    direction_vector = peduncle_point - centroid
    normalized_vector = direction_vector / np.linalg.norm(direction_vector)
    return normalized_vector, centroid, peduncle_point

# Calculating Centroid Line 
direction, centroid, peduncle_point = compute_line(source)
line_points = [centroid, peduncle_point]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(line_points)
line_set.lines = o3d.utility.Vector2iVector([[0, 1]])


# 5. Model Alignment Functions
def compute_rotation(v1, v2):
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Compute rotation axis
    rotation_axis = np.cross(v1, v2)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Compute rotation angle
    cos_angle = np.dot(v1, v2)
    rotation_angle = np.arccos(cos_angle)
    
    return rotation_axis, rotation_angle

def axis_angle_to_rotation_matrix(axis, angle):
    # Using the Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def apply_rotation(pcd, rotation_matrix):
    return np.dot(pcd, rotation_matrix.T)  # .T is for transpose

# -- Consider adding model rotation based on center line. 
def rotation_matrix_around_y(angle_rad):
    return np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

def rotation_matrix_around_x(angle_rad):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])

def rotation_matrix_around_z(angle_rad):
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
# 6. Scaling the model 
# The name of the flipped model is "flipped_pcd"
def get_dimensions(pcd):
    bounding_box = pcd.get_axis_aligned_bounding_box()
    return bounding_box.get_extent()


def scale_point_cloud(source_pcd, target_dimensions, source_dimensions=None):
    if source_dimensions is None:
        source_dimensions = get_dimensions(source_pcd)
    
    scale_factors = [
        target_dimensions[i] / source_dimensions[i]
        for i in range(3)
    ]

    scaled_points = [
        [scale_factors[j] * pt[j] for j in range(3)]
        for pt in source_pcd.points
    ]

    scaled_pcd = o3d.geometry.PointCloud()
    scaled_pcd.points = o3d.utility.Vector3dVector(scaled_points)
    return scaled_pcd

initial_direction = direction / np.linalg.norm(direction) # Direction of the 3D model
rotated_pcd = o3d.geometry.PointCloud()
source_points = np.asarray(source.points)

# 7. Pose registration

def draw_registration_result(source, target, transformation):
  source_temp = copy.deepcopy(source)
  target_temp = copy.deepcopy(target)
  source_temp.paint_uniform_color([1, 0.706, 0])
  target_temp.paint_uniform_color([0, 0.651, 0.929])
  source_temp.transform(transformation)
  o3d.visualization.draw_geometries([source_temp, target_temp])
  # Note: Since the functions "transform" and "paint_uniform_color" change the point cloud,
  # we call copy.deep to make copies and protect the original point clouds.

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size of %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with  search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, 
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source, target):
    print(":: Load two point clouds and disturb the initial pose.")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], 
                             [1.0, 0.0, 0.0, 0.0], 
                             [0.0, 1.0, 0.0, 0.0], 
                             [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds. ")
    print("   Since the downsampling voxel sixe is %.3f, " % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, 
        distance_threshold, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            )
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result

voxel_size = 0.005 # means 5cm for the original dataset... gotta check on mine

def main():

    # ------------ 3D PC Visualization Initialization ------------
    # Load camera intrinsics
    with open('intrinsic.json', 'r') as f:
        intrinsic_json = json.load(f)
    
    # Convert flat list to 3x3 nested list
    intrinsic_matrix_flat = intrinsic_json['intrinsic_matrix']
    intrinsic_matrix = [
        intrinsic_matrix_flat[0:3],
        intrinsic_matrix_flat[3:6],
        intrinsic_matrix_flat[6:9],
    ]
    intrinsic_matrix = np.array(intrinsic_matrix)
    # ------------------------------------------------------


    # Kinect Configurations
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        )
    )
    k4a.start()

    # -------------- 3D PC Init. Variables -------------

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window()
    pcd = o3d.geometry.PointCloud()
    # --------------------------------------------------

    try: 
        while True: 
            capture = k4a.get_capture()

            # Color image operations
            if capture.color is not None:
                #color_image_blurred = cv2.GaussianBlur(capture.color, (7, 7), 0)
                #filtered_color_mask = filter_color(capture.color, lower_green, upper_green)
                 filtered_color_mask = filter_color(capture.color, lower_green, upper_green)
                #cv2.imshow("Filtered color mask", filtered_color_mask)
                #cv2.imshow("Color", capture.color)
            
            # Depth image operations
            if capture.transformed_depth is not None:
                depth_filtered_mask = filter_depth(capture.transformed_depth, MIN_DEPTH, MAX_DEPTH)
                #cv2.imshow("Depth mask", depth_filtered_mask)                
                #cv2.imshow("Depth Image", colorize(capture.transformed_depth, (MIN_DEPTH, MAX_DEPTH)))
            
            # Fused mask operations
            fused_mask = cv2.bitwise_and(filtered_color_mask, depth_filtered_mask)
            #cv2.imshow("Fused mask", fused_mask)
            # Getting the connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fused_mask, 4, cv2.CV_32S)
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_label = np.argmax(areas) + 1  # Add 1 to skip the background
            largest_component = np.zeros_like(labels, dtype=np.uint8)
            largest_component[labels == max_label] = 255

            # Applying the connected component mask
            result_color = cv2.bitwise_and(capture.color, capture.color, mask=largest_component)
            result_depth = cv2.bitwise_and(capture.transformed_depth, capture.transformed_depth, mask=largest_component)

            # Draw circle @ the centroid 
            centroid = tuple(int(c) for c in centroids[max_label])
            result_color_with_mark = cv2.circle(result_color, centroid, 10, (0, 255, 0), -1)
            capture_color_with_mark = capture.color.copy()
            cv2.circle(capture_color_with_mark, centroid, 10, (0,255,0), -1)
            
            #z = capture.transformed_depth[centroid[1], centroid[0]]
            #trajectory.append((centroid[0], centroid[1], z))

            # Live point cloud visualization
            points = get_point_cloud_vectorized(result_depth, intrinsic_matrix=intrinsic_matrix)
            pcd.points = o3d.utility.Vector3dVector(points)

            pcd.transform(transformation_matrix)
            # ----------------------------------------------------------------------------
            # Compute centroid
            centroid = np.mean(np.asarray(pcd.points), axis=0)
            # Translate point cloud to make the centroid the origin
            translated_points = np.asarray(pcd.points) - centroid
            pcd.points = o3d.utility.Vector3dVector(translated_points)
            # Information to orient and scale point cloud and model
            direction_scan, centroid_scan, peduncle_point_scan = compute_line_scan(pcd)
            line_points_scan = [centroid_scan, peduncle_point_scan]  
            line_set_scan = o3d.geometry.LineSet()
            line_set_scan.points = o3d.utility.Vector3dVector(line_points)
            line_set_scan.lines = o3d.utility.Vector2iVector([[0, 1]])
            target_direction = direction_scan / np.linalg.norm(direction_scan)
            # Compute the rotation
            axis, angle = compute_rotation(initial_direction, target_direction)
            R = axis_angle_to_rotation_matrix(axis, angle)

            

            # Apply rotation to 3D model
            rotated_points_np = apply_rotation(source_points, R)
            rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points_np)
            
            # ------ CODE TO REALIGN BASED ON CENTER LINE -------
            angle_180_degrees = (np.pi)  # 180° in radians
            R_180 = rotation_matrix_around_z(angle_180_degrees)
            flipped_points_np = apply_rotation(rotated_points_np, R_180)

            flipped_pcd = o3d.geometry.PointCloud()
            flipped_pcd.points = o3d.utility.Vector3dVector(flipped_points_np)
            # ---------------------------------------------------


            # Scale the model. pcd = kinect scan. rotated_pcd = model. 
            target_dimensions = get_dimensions(pcd)
            scaled_pcd2 = scale_point_cloud(flipped_pcd, target_dimensions) # This is my model. 
            # We paint the point clouds to be able to distinguish between them
            pcd.paint_uniform_color([1, 0, 0])  # Paint the first point cloud red
            scaled_pcd2.paint_uniform_color([0, 1, 0])  # Paint the scaled point cloud green
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size=voxel_size, source=scaled_pcd2, target=pcd)
            result_ransac = execute_global_registration(source_down, target_down, 
                                            source_fpfh, target_fpfh, 
                                            voxel_size)
            print(result_ransac)
            # ----------------------------------------------------------------------------
            vis.clear_geometries()
            vis.add_geometry(scaled_pcd2) # Variables and their meaning in this line. pcd = kinect scan. rotated_pcd = rotated scan. It was declared outside but values are assigned in main loop.
            vis.update_geometry(scaled_pcd2)
            vis.poll_events()
            vis.update_renderer()

            vis.clear_geometries()
            vis2.add_geometry(pcd)
            vis2.update_geometry(pcd)
            vis2.poll_events()
            vis2.update_renderer()

            #cv2.imshow("Result color", result_color_with_mark)
            #cv2.imshow("Result Depth", colorize(result_depth, (MIN_DEPTH, MAX_DEPTH)))
            cv2.imshow("Original image with centroid", capture_color_with_mark)

            key = cv2.waitKey(10)
            if key != -1:
                #cv2.destroyAllWindows()
                break
    finally: 
        cv2.destroyAllWindows()
        vis.destroy_window()
        k4a.stop()
    #plot_thread.join()

if __name__ == "__main__":
    main()
