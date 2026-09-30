#!/usr/bin/env python3
"""
ZED2i Plane Visualizer  –  RGB Camera Overlay (AR mode)
=========================================================
Subscribes to:
  /zed/zed_node/rgb/color/rect/image          – live left RGB feed
  /zed/zed_node/rgb/color/rect/camera_info    – camera intrinsics (K matrix)
  /zed/zed_node/point_cloud/cloud_registered  – 3-D point cloud

FOV applied before sampling:
  depth  (X in zed_left_camera_frame) : 0.1 – 3.0 m
  lateral(Y)                          : ±0.75 m  →  1.5 m wide

Three spatially-spread 3-D points are chosen from the FOV cloud.
Their plane is projected and drawn on the live RGB image:
  • Filled semi-transparent amber quadrilateral  (the plane)
  • Coloured circles at each of the 3 sample points
  • Coordinate labels (x, y, z in metres)
  • HUD panel with plane-equation and normal vector

Usage
-----
  source /opt/ros/humble/setup.bash
  python3 plane_visualizer.py
"""

import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from collections import deque

# ── Dense plane fitting ────────────────────────────────────────────────
GRID_M            = 10       # depth cells for grid sampling
GRID_N            = 10       # lateral cells for grid sampling
RANSAC_ITERATIONS = 60       # RANSAC iterations
RANSAC_THRESH     = 0.03     # inlier distance threshold in metres
MIN_INLIERS       = 20       # minimum inliers to trust RANSAC result
DEPTH_BIN_WIDTH   = 0.15     # histogram bin width for depth isolation
NORMAL_BUF_SIZE   = 10       # smoothing buffer length

# ── Slope thresholds (degrees) ─────────────────────────────────────────
THR_FLAT          = 10.0
THR_GENTLE        = 25.0
THR_STEEP         = 45.0
THR_EXTREME       = 75.0

# ── Confidence gates ───────────────────────────────────────────────────
MIN_PLANARITY     = 0.50
MIN_INLIER_RATIO  = 0.25

# ─────────────────────────── tunable FOV ──────────────────────────────────
FOV_DEPTH_MAX = 3.0    # metres ahead  (X axis in zed_left_camera_frame)
FOV_LAT_HALF  = 0.75   # ± metres laterally (Y axis)
# ──────────────────────────────────────────────────────────────────────────

# BGR colours (OpenCV order)
COL_PLANE  = (0,  180, 255)   # amber
COL_P = [
    (60,  60,  255),           # P1 red
    (60,  220,  60),           # P2 green
    (255, 100,  60),           # P3 blue
]
COL_NORMAL = (0, 240, 240)    # cyan
COL_HUD_BG = (15,  15,  15)
COL_HUD_FG = (220, 220, 220)

PLANE_HALF_SIZE = 0.9   # half-size of drawn plane square in metres


# ──────────────────────────── helpers ─────────────────────────────────────

def grid_sample_fov(pts: np.ndarray, depth_col: int, lat_col: int, grid_m=GRID_M, grid_n=GRID_N) -> np.ndarray:
    if pts.shape[0] == 0:
        return np.empty((0, 3))
    d_min, d_max = pts[:, depth_col].min(), pts[:, depth_col].max()
    l_min, l_max = pts[:, lat_col].min(), pts[:, lat_col].max()
    d_step = (d_max - d_min) / grid_m
    l_step = (l_max - l_min) / grid_n
    
    sampled = []
    # add small epsilon to avoid out of bounds on exactly max value
    d_idx = np.clip(((pts[:, depth_col] - d_min) / (d_step + 1e-9)).astype(int), 0, grid_m - 1)
    l_idx = np.clip(((pts[:, lat_col] - l_min) / (l_step + 1e-9)).astype(int), 0, grid_n - 1)
    
    for i in range(grid_m):
        for j in range(grid_n):
            mask = (d_idx == i) & (l_idx == j)
            if np.any(mask):
                cell_pts = pts[mask]
                # median point in cell based on depth
                d_med = np.median(cell_pts[:, depth_col])
                idx = np.argmin(np.abs(cell_pts[:, depth_col] - d_med))
                sampled.append(cell_pts[idx])
    return np.array(sampled)

def fit_plane_svd(pts: np.ndarray, depth_col: int):
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n = Vt[-1]
    
    if n[depth_col] < 0:
        n = -n
        
    planarity_score = 1.0 - (S[2] / S[0]) if S[0] > 1e-9 else 0.0
    d = float(np.dot(n, centroid))
    return n, d, centroid, planarity_score

def ransac_plane(pts: np.ndarray, depth_col: int, n_iter=RANSAC_ITERATIONS, inlier_thresh=RANSAC_THRESH):
    N = pts.shape[0]
    if N < 3:
        return None
        
    best_inlier_count = 0
    best_inlier_mask = None
    
    for _ in range(n_iter):
        idxs = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = pts[idxs]
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        
        d = np.dot(n, p1)
        # distance of all points to plane
        dists = np.abs(np.dot(pts, n) - d)
        inlier_mask = dists < inlier_thresh
        inlier_count = np.sum(inlier_mask)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            
    if best_inlier_count < MIN_INLIERS:
        return None
        
    # Refit SVD on inliers
    inliers = pts[best_inlier_mask]
    n, d, centroid, planarity = fit_plane_svd(inliers, depth_col)
    
    # re-evaluate mask with refined plane (optional, but good practice)
    dists = np.abs(np.dot(pts, n) - d)
    refined_mask = dists < inlier_thresh
    return n, d, centroid, planarity, refined_mask

def find_dominant_depth(pts: np.ndarray, depth_col: int, bin_width=DEPTH_BIN_WIDTH):
    if pts.shape[0] == 0:
        return np.ones(0, dtype=bool)
    
    depths = pts[:, depth_col]
    d_min, d_max = depths.min(), depths.max()
    bins = np.arange(d_min, d_max + bin_width, bin_width)
    if len(bins) < 2:
        return np.ones(pts.shape[0], dtype=bool)
        
    hist, edges = np.histogram(depths, bins=bins)
    peak_idx = np.argmax(hist)
    peak_center = (edges[peak_idx] + edges[peak_idx+1]) / 2.0
    
    mask = np.abs(depths - peak_center) <= bin_width
    return mask

def compute_slope(n: np.ndarray, depth_col: int):
    # n_ref horizontal normal faces UP
    if depth_col == 2: # optical: Y is down, so UP is -Y
        n_ref = np.array([0.0, -1.0, 0.0])
    else: # zed_left: Z is up
        n_ref = np.array([0.0, 0.0, 1.0])
        
    # Enforce normal faces "UP" relative to horizontal, so angle never > 90
    if np.dot(n, n_ref) < 0:
        n = -n
        
    cos_theta = np.clip(np.dot(n_ref, n), -1.0, 1.0)
    theta_deg = float(np.degrees(np.arccos(cos_theta)))
    
    # pitch: tilt forward/back. Compare forward vs up component
    # forward is depth_col.
    pitch_deg = float(np.degrees(np.arctan2(n[depth_col], np.dot(n, n_ref))))
    
    # camber: tilt left/right.
    # lateral is 0 if optical (X right), 1 if zed_left (Y left)
    lat_col = 0 if depth_col == 2 else 1
    camber_deg = float(np.degrees(np.arctan2(n[lat_col], np.dot(n, n_ref))))
    
    if theta_deg <= THR_FLAT:
        cls, col = "FLAT", (0, 200, 0)
    elif theta_deg <= THR_GENTLE:
        cls, col = "GENTLE SLOPE", (0, 200, 255)
    elif theta_deg <= THR_STEEP:
        cls, col = "STEEP SLOPE", (0, 140, 255)
    elif theta_deg <= THR_EXTREME:
        cls, col = "EXTREME SLOPE", (0, 0, 255)
    else:
        cls, col = "WALL / OBSTACLE", (0, 0, 255)
        
    return {
        "theta": theta_deg,
        "pitch": pitch_deg,
        "camber": camber_deg,
        "classification": cls,
        "colour": col
    }

def three_fixed_points(pts: np.ndarray, lat_col: int = 1) -> np.ndarray:
    """
    Pick 3 points whose (depth, lateral) coordinates are closest to a fixed
    equilateral triangle pattern.
    P1: 0.5m ahead, 0.0m lateral
    P2: 1.0m ahead, -0.3m lateral
    P3: 1.0m ahead, +0.3m lateral
    """
    depth_col = 0 if lat_col != 0 else 2
    
    targets = [
        (0.5, 0.0),    # P1 (top/closest)
        (1.0, -0.3),   # P2 (bottom left)
        (1.0, 0.3)     # P3 (bottom right)
    ]
    
    selected = []
    for (t_depth, t_lat) in targets:
        dist_sq = (pts[:, depth_col] - t_depth)**2 + (pts[:, lat_col] - t_lat)**2
        # Average points within 5cm radius for stability against noise
        min_dist = np.min(dist_sq)
        if min_dist < 0.1**2:
            mask = dist_sq < (np.sqrt(min_dist) + 0.05)**2
            pt = pts[mask].mean(axis=0)
            selected.append(pt)
        else:
            idx = np.argmin(dist_sq)
            selected.append(pts[idx])
            
    return np.array(selected)


def plane_from_3pts(p1, p2, p3):
    """Return (unit_normal, d) for plane n·x = d.  Normal faces camera."""
    v1, v2 = p2 - p1, p3 - p1
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0., 0., 1.]), 0.0
    n /= norm
    if n[0] < 0:      # ensure normal points toward camera (+X direction)
        n = -n
    return n, float(np.dot(n, p1))


def camera_to_optical(pts: np.ndarray) -> np.ndarray:
    """
    Transform points from zed_left_camera_frame  to optical frame.
    camera_frame : X forward, Y left,  Z up
    optical_frame: X right,  Y down,  Z forward
      optical_x = -camera_y
      optical_y = -camera_z
      optical_z =  camera_x
    """
    out = np.empty_like(pts)
    out[:, 0] = -pts[:, 1]   # right  = -left
    out[:, 1] = -pts[:, 2]   # down   = -up
    out[:, 2] =  pts[:, 0]   # fwd    = fwd
    return out


def project_pts(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """
    Project (N,3) 3-D points in optical frame to (N,2) pixel coords.
    Returns None if any point is behind camera.
    """
    opt = camera_to_optical(pts_cam)
    if np.any(opt[:, 2] <= 0):
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (fx * opt[:, 0] / opt[:, 2] + cx).astype(int)
    v = (fy * opt[:, 1] / opt[:, 2] + cy).astype(int)
    return np.stack([u, v], axis=1)


def plane_corners_3d(pts3: np.ndarray, n: np.ndarray, half: float) -> np.ndarray:
    """Build 4 corners of a square plane centred at the centroid of pts3."""
    if len(pts3) == 0:
        return np.empty((0, 3))
    centroid = pts3.mean(axis=0)
    # tangent vectors in the plane
    ref = np.array([0., 0., 1.])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0., 1., 0.])
    t1 = np.cross(n, ref);  t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1);   t2 /= np.linalg.norm(t2)
    return np.array([
        centroid + half * t1 + half * t2,
        centroid - half * t1 + half * t2,
        centroid - half * t1 - half * t2,
        centroid + half * t1 - half * t2,
    ])


def decode_image(msg: Image) -> np.ndarray | None:
    """Convert a ROS Image (rgb8 / bgr8) to a BGR OpenCV Mat."""
    h, w = msg.height, msg.width
    enc  = msg.encoding.lower()
    arr  = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8", "8uc3"):
        img = arr.reshape(h, w, 3)
        return img[..., ::-1].copy() if enc == "rgb8" else img.copy()
    if enc in ("rgba8", "bgra8"):
        img = arr.reshape(h, w, 4)
        img = img[..., :3]
        return img[..., ::-1].copy() if enc == "rgba8" else img.copy()
    if enc in ("mono8", "8uc1"):
        img = arr.reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return None


# ──────────────────────────── ROS2 Node ───────────────────────────────────

class PlaneVisualizerNode(Node):

    def __init__(self):
        super().__init__("plane_visualizer")
        self._lock   = threading.Lock()
        self._latest = None       # dict with plane/points data
        self._img    = None       # latest BGR OpenCV image
        self._K      = None       # 3×3 camera intrinsic matrix
        self._new    = threading.Event()
        
        self._normal_buf = deque(maxlen=NORMAL_BUF_SIZE)
        self._frame_count = 0

        qos = rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value

        self.sub_cloud = self.create_subscription(
            PointCloud2, "/zed/zed_node/point_cloud/cloud_registered",
            self._cloud_cb, qos)
        self.sub_img   = self.create_subscription(
            Image, "/zed/zed_node/rgb/color/rect/image",
            self._img_cb, qos)
        self.sub_info  = self.create_subscription(
            CameraInfo, "/zed/zed_node/rgb/color/rect/camera_info",
            self._info_cb, qos)

        self.get_logger().info(
            "Subscribed to RGB image + camera_info + point_cloud. Waiting…"
        )

    # ── camera intrinsics ────────────────────────────────────────────────
    def _info_cb(self, msg: CameraInfo):
        if self._K is None:
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            with self._lock:
                self._K = K
            self.get_logger().info(
                f"Got camera intrinsics  fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}"
            )

    # ── RGB image ─────────────────────────────────────────────────────────
    def _img_cb(self, msg: Image):
        img = decode_image(msg)
        if img is None:
            self.get_logger().warning(
                f"Cannot decode image encoding '{msg.encoding}'"
            )
            return
        with self._lock:
            self._img = img
        self._new.set()

    # ── point cloud ───────────────────────────────────────────────────────
    def _cloud_cb(self, msg: PointCloud2):
        try:
            step_f    = msg.point_step // 4
            total_pts = msg.width * msg.height
            raw = np.frombuffer(msg.data, dtype=np.float32)
            raw = raw[: total_pts * step_f].reshape(-1, step_f)
            xyz = raw[:, :3].copy()
            valid    = np.all(np.isfinite(xyz), axis=1)
            pts_raw  = xyz[valid]
        except Exception as e:
            self.get_logger().error(f"Point cloud parse error: {e}")
            return

        if pts_raw.shape[0] < 100:
            return

        # One-shot frame debug
        if not hasattr(self, "_dbg"):
            self._dbg = True
            mn, mx = pts_raw.min(0), pts_raw.max(0)
            self.get_logger().info(
                f"[FRAME DEBUG] x=[{mn[0]:.2f},{mx[0]:.2f}] "
                f"y=[{mn[1]:.2f},{mx[1]:.2f}] "
                f"z=[{mn[2]:.2f},{mx[2]:.2f}] "
                f"frame='{msg.header.frame_id}'"
            )

        # Frame detection: optical → Z depth; else → X depth
        frame = msg.header.frame_id
        if "optical" in frame:
            depth_col, lat_col = 2, 0
        else:
            depth_col, lat_col = 0, 1   # zed_left_camera_frame

        # FOV filter
        mask = (
            (pts_raw[:, depth_col] > 0.1) &
            (pts_raw[:, depth_col] < FOV_DEPTH_MAX) &
            (pts_raw[:, lat_col]   > -FOV_LAT_HALF) &
            (pts_raw[:, lat_col]   <  FOV_LAT_HALF)
        )
        pts = pts_raw[mask]
        if pts.shape[0] < 30:
            return

        # 1. find dominant depth
        dom_mask = find_dominant_depth(pts, depth_col)
        pts_dom = pts[dom_mask]
        
        # 2. grid sample
        sampled_pts = grid_sample_fov(pts_dom, depth_col, lat_col)
        
        # 3. RANSAC
        ransac_result = ransac_plane(sampled_pts, depth_col)
        
        if ransac_result is not None:
            n, d, centroid, planarity, inlier_mask = ransac_result
        else:
            # Fallback to fixed points
            pts3 = three_fixed_points(pts_dom, lat_col=lat_col)
            n, d = plane_from_3pts(pts3[0], pts3[1], pts3[2])
            centroid = pts3.mean(axis=0)
            planarity = 0.0
            inlier_mask = np.ones(3, dtype=bool)
            sampled_pts = pts3
            
        # smoothing buffer
        self._normal_buf.append(n)
        smoothed_n = np.mean(self._normal_buf, axis=0)
        smoothed_n /= np.linalg.norm(smoothed_n)
        
        slope = compute_slope(smoothed_n, depth_col)
        
        # for drawing plane corners
        corners = plane_corners_3d(sampled_pts, smoothed_n, PLANE_HALF_SIZE)
        
        candidate = {
            "n": smoothed_n, 
            "d": float(np.dot(smoothed_n, centroid)),
            "corners": corners, 
            "frame": frame,
            "slope": slope,
            "sampled_pts": sampled_pts,
            "inlier_mask": inlier_mask,
            "planarity": planarity,
            "centroid": centroid,
        }

        with self._lock:
            self._latest = candidate
            self._frame_count += 1
            candidate["frame_count"] = self._frame_count

        self._new.set()


# ──────────────────────────── OpenCV Renderer ─────────────────────────────

def draw_overlay(img: np.ndarray, data: dict, K: np.ndarray) -> np.ndarray:
    """Render AR overlay on img and return it."""
    out    = img.copy()
    h, w   = out.shape[:2]
    
    n = data["n"]
    d = data["d"]
    corners = data["corners"]
    slope = data["slope"]
    sampled_pts = data["sampled_pts"]
    inlier_mask = data["inlier_mask"]
    planarity = data["planarity"]
    fc = data.get("frame_count", 0)
    
    col_plane = slope["colour"]
    inlier_ratio = np.sum(inlier_mask) / max(1, len(inlier_mask))
    
    is_confident = (planarity >= MIN_PLANARITY) and (inlier_ratio >= MIN_INLIER_RATIO)
    
    if not is_confident:
        txt = "UNCERTAIN - INSUFFICIENT SURFACE"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(out, txt, ((w - tw) // 2, (h + th) // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2, cv2.LINE_AA)
    else:
        # ── project plane corners ──────────────────────────────────────────────
        corner_px = project_pts(corners, K)
        if corner_px is not None:
            overlay = out.copy()
            cv2.fillPoly(overlay, [corner_px.reshape(-1, 1, 2)], col_plane)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
            cv2.polylines(out, [corner_px.reshape(-1, 1, 2)],
                          True, col_plane, 2, cv2.LINE_AA)

    # ── project and draw sampled points ──────────────────────────────────
    pxs = project_pts(sampled_pts, K)
    if pxs is not None:
        for i, px in enumerate(pxs):
            u, v = int(px[0]), int(px[1])
            if not (0 <= u < w and 0 <= v < h): 
                continue
            
            is_inlier = inlier_mask[i] if i < len(inlier_mask) else True
            if is_inlier:
                cv2.circle(out, (u, v), 4, (0, 200, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(out, (u, v), 3, (0, 0, 255), -1, cv2.LINE_AA)

    # flashing border for obstacles
    if slope["classification"] == "WALL / OBSTACLE" and (fc % 4) < 2:
        cv2.rectangle(out, (0, 0), (w-1, h-1), (0, 0, 255), 8)

    # ── HUD panel (top-left) ───────────────────────────────────────────────
    hud_lines = [
        "ZED2i  Plane Visualizer",
        f"FOV: depth < {FOV_DEPTH_MAX:.1f}m  lat +/-{FOV_LAT_HALF:.2f}m",
        "",
        f"Normal  nx={n[0]:+.3f}  ny={n[1]:+.3f}  nz={n[2]:+.3f}",
        f"Plane eq:  {n[0]:+.3f}x {n[1]:+.3f}y {n[2]:+.3f}z = {d:+.3f}",
        "----------------------"
    ]
    
    if planarity < MIN_PLANARITY:
        pass # we handle slope separately below
    else:
        pass
        
    p_dir = "uphill" if slope['pitch'] > 0 else "downhill"
    hud_lines.append(f"Pitch:  {slope['pitch']:+.1f} deg  ({p_dir})")
    
    c_dir = "left lean" if slope['camber'] > 0 else "right lean"
    hud_lines.append(f"Camber: {slope['camber']:+.1f} deg  ({c_dir})")
    
    hud_lines.append(f"Planarity: {planarity:.2f}   Inliers: {np.sum(inlier_mask)}/{len(inlier_mask)}")
    hud_lines.append("----------------------")
    hud_lines.append("")
    hud_lines.append("[Q] quit")

    line_h = 20
    pad    = 8
    hud_w  = 380
    hud_h  = len(hud_lines) * line_h + pad * 2

    hud_bg          = out[pad: pad + hud_h, pad: pad + hud_w].copy()
    darken          = np.zeros_like(hud_bg)
    hud_bg          = cv2.addWeighted(hud_bg, 0.3, darken, 0.7, 0)
    out[pad: pad + hud_h, pad: pad + hud_w] = hud_bg

    for i, line in enumerate(hud_lines):
        col  = (255, 190, 50) if i == 0 else COL_HUD_FG
                
        size = 0.50 if i == 0 else 0.42
        cv2.putText(out, line,
                    (pad * 2, pad + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, size, col, 1, cv2.LINE_AA)

    # ── Big slope readout (bottom-left) ────────────────────────────────────
    if planarity < MIN_PLANARITY:
        slope_txt = "SLOPE: LOW CONFIDENCE"
        slope_col = (0, 140, 255) # orange
    else:
        slope_txt = f"SLOPE: {slope['theta']:.1f}   {slope['classification']}"
        slope_col = slope["colour"]

    (bw, bh), _ = cv2.getTextSize(slope_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    bx, by = 20, h - 30
    
    # dark background for big text
    overlay = out.copy()
    cv2.rectangle(overlay, (bx - 10, by - bh - 10), (bx + bw + 10, by + 10), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
    
    cv2.putText(out, slope_txt, (bx, by), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, slope_col, 3, cv2.LINE_AA)

    return out


# ──────────────────────────── Main loop ───────────────────────────────────

def main():
    rclpy.init()
    node = PlaneVisualizerNode()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    cv2.namedWindow("ZED2i Plane Visualizer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZED2i Plane Visualizer", 1280, 720)

    print("Window open.  Q / Esc = quit")

    while True:
        # Wake up on image OR plane update (max 50 ms wait)
        node._new.wait(timeout=0.05)
        node._new.clear()

        with node._lock:
            img    = node._img.copy() if node._img is not None else None
            data   = node._latest
            K      = node._K

        if img is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for RGB image…", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180),
                        1, cv2.LINE_AA)
            cv2.imshow("ZED2i Plane Visualizer", blank)
        elif data is None or K is None:
            waiting = img.copy()
            msg = "Waiting for camera_info…" if K is None else "Searching for plane…"
            cv2.putText(waiting, msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255),
                        1, cv2.LINE_AA)
            cv2.imshow("ZED2i Plane Visualizer", waiting)
        else:
            rendered = draw_overlay(img, data, K)
            cv2.imshow("ZED2i Plane Visualizer", rendered)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):          # Q / Esc  → quit
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.try_shutdown()
    spin_thread.join(timeout=2)
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
