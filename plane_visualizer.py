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

        pts3     = three_fixed_points(pts, lat_col=lat_col)
        
        # Temporal smoothing for rigidity
        with self._lock:
            if not hasattr(self, '_smooth_pts'):
                self._smooth_pts = pts3.copy()
            else:
                self._smooth_pts = 0.1 * pts3 + 0.9 * self._smooth_pts
            smoothed = self._smooth_pts.copy()
            
        n, d     = plane_from_3pts(smoothed[0], smoothed[1], smoothed[2])
        corners  = plane_corners_3d(smoothed, n, PLANE_HALF_SIZE)

        candidate = {
            "pts3": smoothed, "n": n, "d": d,
            "corners": corners, "frame": frame,
        }

        with self._lock:
            self._latest = candidate

        self._new.set()


# ──────────────────────────── OpenCV Renderer ─────────────────────────────

def draw_overlay(img: np.ndarray, data: dict, K: np.ndarray) -> np.ndarray:
    """Render AR overlay on img and return it."""
    out    = img.copy()
    h, w   = out.shape[:2]
    pts3   = data["pts3"]
    n      = data["n"]
    d      = data["d"]
    corners = data["corners"]

    # ── project plane corners ──────────────────────────────────────────────
    corner_px = project_pts(corners, K)
    if corner_px is not None:
        # clip to image bounds
        ok = np.all((corner_px[:, 0] >= -w) & (corner_px[:, 0] < 2 * w) &
                    (corner_px[:, 1] >= -h) & (corner_px[:, 1] < 2 * h), axis=0)
        if ok:
            overlay = out.copy()
            cv2.fillPoly(overlay, [corner_px.reshape(-1, 1, 2)], COL_PLANE)
            cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)   # make square more faint
            cv2.polylines(out, [corner_px.reshape(-1, 1, 2)],
                          True, COL_PLANE, 1, cv2.LINE_AA)

    # ── highlight plane specifically through the 3 points (triangle) ───────
    pxs = project_pts(pts3, K)
    if pxs is not None:
        triangle_pts = pxs.reshape(-1, 1, 2).astype(np.int32)
        
        # draw filled semi-transparent triangle
        overlay = out.copy()
        cv2.fillPoly(overlay, [triangle_pts], COL_PLANE)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
        
        # draw thick glowing border connecting P1 -> P2 -> P3 -> P1
        cv2.polylines(out, [triangle_pts], True, (0, 240, 255), 3, cv2.LINE_AA)

    # ── project and draw 3 sample points ──────────────────────────────────
    labels = ["P1", "P2", "P3"]
    if pxs is not None:
        for i, (px, pt, col) in enumerate(zip(pxs, pts3, COL_P)):
            u, v = int(px[0]), int(px[1])
            if not (0 <= u < w and 0 <= v < h):
                continue
            cv2.circle(out, (u, v), 14, col, -1, cv2.LINE_AA)
            cv2.circle(out, (u, v), 14, (255, 255, 255), 2, cv2.LINE_AA)
            # label
            txt = f"{labels[i]}  ({pt[0]:+.2f},{pt[1]:+.2f},{pt[2]:+.2f})m"
            tx, ty = u + 18, v + 6
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(out, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3),
                          (20, 20, 20), -1)
            cv2.putText(out, txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

    # ── HUD panel (top-left) ───────────────────────────────────────────────
    hud_lines = [
        "ZED2i  Plane Visualizer",
        f"FOV: depth < {FOV_DEPTH_MAX:.1f}m  lat \xb1{FOV_LAT_HALF:.2f}m",
        "",
        f"Normal  nx={n[0]:+.3f}  ny={n[1]:+.3f}  nz={n[2]:+.3f}",
        f"Plane eq:  {n[0]:+.3f}x {n[1]:+.3f}y {n[2]:+.3f}z = {d:+.3f}",
        "",
    ]
    if pxs is not None:
        for i, pt in enumerate(pts3):
            hud_lines.append(
                f"  {labels[i]}: x={pt[0]:+.3f} y={pt[1]:+.3f} z={pt[2]:+.3f}"
            )
    hud_lines.append("")
    hud_lines.append("[Q] quit")

    line_h = 20
    pad    = 8
    hud_w  = 360
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
