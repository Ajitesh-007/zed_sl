# ZED2i Plane Visualizer (AR Overlay)

## Overview
This project is a ROS2 (Humble) node that overlays a dynamically computed 3D plane onto a live RGB camera feed from a **Stereolabs ZED2i** camera. It is designed for a rover to visualize the ground slope directly ahead of it in real-time. 

Instead of showing a raw 3D point cloud, it acts as an Augmented Reality (AR) Heads-Up Display (HUD) using OpenCV. It computes a plane from three specific, fixed coordinate locations ahead of the rover and draws a semi-transparent amber plane, the points themselves, and the normal vector data directly onto the live RGB image.

## ROS2 Architecture

**Node Name:** `plane_visualizer`

### Subscribed Topics
- **`/zed/zed_node/rgb/color/rect/image`** (`sensor_msgs/Image`): 
  The rectified left RGB camera feed. Decoded into a BGR numpy array for OpenCV rendering.
- **`/zed/zed_node/rgb/color/rect/camera_info`** (`sensor_msgs/CameraInfo`): 
  Provides the camera intrinsics (K matrix) required to project 3D coordinate points back into the 2D image plane.
- **`/zed/zed_node/point_cloud/cloud_registered`** (`sensor_msgs/PointCloud2`): 
  The dense 3D point cloud aligned with the left RGB camera.

## Core Logic & Technical Implementation

### 1. Fast Point Cloud Parsing
The standard `sensor_msgs_py.point_cloud2` parser can be slow and struggles with certain structured array conversions in newer NumPy versions. This script bypasses it by reading the raw byte buffer using `np.frombuffer(msg.data, dtype=np.float32)`, restructuring it based on `point_step`, and isolating the X, Y, Z float coordinates for extremely fast processing.

### 2. Coordinate Frames & Transformations
The ZED wrapper can output point clouds in either the base camera link frame or the optical frame. The code dynamically detects this via the `msg.header.frame_id`:
- **`zed_left_camera_frame` (Standard ROS)**: X is depth (forward), Y is lateral (left), Z is height (up).
- **`optical_frame`**: Z is depth (forward), X is lateral (right), Y is height (down).

Before projection, all 3D points are converted to the standard Optical Frame so they can be multiplied by the Intrinsic K Matrix.

### 3. Fixed Target Point Selection
To prevent the plane from jumping around due to point cloud noise, the node looks for 3 fixed geometric locations forming an equilateral triangle in front of the rover. 
- **P1**: 0.5m ahead, 0.0m lateral
- **P2**: 1.0m ahead, -0.3m lateral
- **P3**: 1.0m ahead, +0.3m lateral

For each target, it calculates the 2D Euclidean distance (depth and lateral axes) of all point cloud points to the target. To ensure Z-height stability, it takes all points within a 5cm radius of the target and averages them.

### 4. Plane Computation
Using the three stable 3D points, the script computes the plane's normal vector using a cross product ($v_1 \times v_2$). The normal vector is normalized and forced to face the camera. The plane equation ($n \cdot x = d$) is derived and displayed.

### 5. AR Projection & OpenCV Rendering
Using the projection formula $u = f_x \frac{X}{Z} + c_x$ and $v = f_y \frac{Y}{Z} + c_y$, the 3D plane corners and the 3 sample points are projected onto the 2D image plane.
- `cv2.fillPoly` and `cv2.addWeighted` are used to draw a semi-transparent amber plane on the ground.
- A HUD (Heads-Up Display) is drawn in the top-left corner detailing the normal vector, plane equation, and coordinate exacts for P1, P2, and P3. 

## Requirements
- ROS2 Humble
- ZED ROS2 Wrapper (`zed_wrapper`)
- Python Packages: `numpy`, `opencv-python` (cv2), `rclpy`
