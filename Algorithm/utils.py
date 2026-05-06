# -*- coding: utf-8 -*-
"""
Algorithm.utils — 图像处理基础工具
"""
import cv2
import numpy as np
from Algorithm import Config


# ====================================================================
#  角点几何工具
# ====================================================================
def order_corners(pts):
    """将4个点排成 左上/右上/右下/左下 顺序"""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def find_4corners(cnt):
    """从轮廓中提取4个角点，失败返回 None"""
    peri = cv2.arcLength(cnt, True)
    for eps in [0.02, 0.04, 0.06, 0.08]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx.astype(np.float32))
    return None


def diagonal_center(corners):
    """两条对角线的交点，作为矩形中心"""
    p0, p1, p2, p3 = corners
    x1, y1 = p0;  x2, y2 = p2
    x3, y3 = p1;  x4, y4 = p3
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    return int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))


def is_clipped(x, y, w, h, frame_h, frame_w, margin=5):
    """判断包围盒是否贴近画面边缘"""
    return (x < margin or y < margin or
            x + w > frame_w - margin or
            y + h > frame_h - margin)


def is_quad_valid(corners, thresh=0.5):
    """四边形面积 / 外接矩形面积，过小说明形状不规则"""
    quad_area = cv2.contourArea(corners.reshape(4, 1, 2).astype(np.int32))
    x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
    if w * h == 0:
        return False
    return (quad_area / (w * h)) > thresh


# ====================================================================
#  透视变换工具
# ====================================================================
def find_corners_for_warp(cnt):
    """从轮廓获取用于透视变换的4角，找不到4边形就用最小外接矩形"""
    corners = find_4corners(cnt)
    if corners is not None:
        return corners
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_corners(box)


def perspective_warp(frame, corners):
    """对 frame 做透视校正，返回 (warped, M, M_inv)"""
    tl, tr, br, bl = corners
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    warp_w = max(int(max(w_top, w_bot) * Config.WARP_SCALE), 10)
    warp_h = max(int(max(h_left, h_right) * Config.WARP_SCALE), 10)
    dst = np.array([[0, 0], [warp_w - 1, 0],
                    [warp_w - 1, warp_h - 1], [0, warp_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    M_inv = cv2.getPerspectiveTransform(dst, corners)
    return cv2.warpPerspective(frame, M, (warp_w, warp_h)), M, M_inv


# ====================================================================
#  红色像素验证
# ====================================================================
def validate_red(roi, lower1=None, upper1=None, lower2=None, upper2=None):
    """统计 ROI 内红色像素数，用于候选筛选"""
    if roi is None or roi.size == 0:
        return 0
    l1 = lower1 if lower1 is not None else Config.RED_LOWER1
    u1 = upper1 if upper1 is not None else Config.RED_UPPER1
    l2 = lower2 if lower2 is not None else Config.RED_LOWER2
    u2 = upper2 if upper2 is not None else Config.RED_UPPER2
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return cv2.countNonZero(
        cv2.bitwise_or(cv2.inRange(hsv, l1, u1),
                       cv2.inRange(hsv, l2, u2))
    )


# ====================================================================
#  霍夫圆检测
# ====================================================================
def run_hough(roi, param2=None, min_r_rat=None, max_r_rat=None, min_dist_rat=None):
    """
    在 ROI 上跑霍夫圆。
      - 先缩 50%，参数比例稳定
      - GaussianBlur(7,7) 去噪
      - HOUGH_GRADIENT_ALT：param2 为 [0,1] 置信度
    返回 (circle_count, circles)，circles 坐标已映射回原图尺寸。
    """
    if roi is None or roi.size == 0:
        return 0, None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = roi.shape[:2]

    scale = 0.8
    dp=1.5  #精度分辨率反比，越大检测越快
    p1=40    #Canny边缘检测阈值 越小边缘越多

    # gsmall = gray + scale down
    gsmall = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))))
    gsmall = cv2.GaussianBlur(gsmall, (5, 5), 0)
    short = min(gsmall.shape[:2])           #使用缩小数据
    if short < 10:
        return 0, None
    p2 = param2 if param2 is not None else Config.HOUGH_PARAM2
    #默认值兜底
    _min_r_rat = min_r_rat if min_r_rat is not None else Config.HOUGH_MIN_R_RAT
    _max_r_rat = max_r_rat if max_r_rat is not None else Config.HOUGH_MAX_R_RAT
    _min_dist_rat = min_dist_rat if min_dist_rat is not None else Config.HOUGH_MIN_DIST_RAT
    #max降噪
    min_r = max(3,          int(short * _min_r_rat))
    max_r = max(min_r + 1,  int(short * _max_r_rat))
    min_dist = max(1,    int(short * _min_dist_rat))    #间距
    circles = cv2.HoughCircles(
        gsmall, cv2.HOUGH_GRADIENT_ALT, dp,
        minDist=min_dist, param1=p1, param2=p2,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return 0, None
    circles = np.round(circles[0]).astype(int)
    circles[:, :2] = (circles[:, :2] / scale).astype(int)
    circles[:, 2]  = (circles[:, 2]  / scale).astype(int)
    return len(circles), circles


# ====================================================================
#  加权中心
# ====================================================================
def weighted_center(circles, offset=(0, 0)):
    """按半径加权平均圆心坐标"""
    if circles is None or len(circles) == 0:
        return None, None
    radii = circles[:, 2].astype(float)
    total = radii.sum()
    if total == 0:
        return None, None
    cx = int(np.round(((circles[:, 0] + offset[0]) * radii).sum() / total))
    cy = int(np.round(((circles[:, 1] + offset[1]) * radii).sum() / total))
    return cx, cy


# ====================================================================
#  椭圆拟合（task6）
# ====================================================================
def _fit_ring_best(roi, bw_px, bh_px,
                   phys_w_cm=21.0, phys_h_cm=29.7,
                   target_r_cm=6.0, tol=0.35, area_min=50):
    """
    在 ROI 红色 mask 上拟合所有椭圆，返回与期望 r=6cm 最接近的
    (cx, cy, mean_r)，失败返回 None。
    """
    if roi is None or roi.size == 0:
        return None
    px_per_cm = (bw_px / phys_w_cm + bh_px / phys_h_cm) / 2.0
    expected_r = target_r_cm * px_per_cm

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, Config.RED_LOWER1, Config.RED_UPPER1),
        cv2.inRange(hsv, Config.RED_LOWER2, Config.RED_UPPER2),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid = [c for c in cnts if len(c) >= 5 and cv2.contourArea(c) > area_min]
    if not valid:
        return None

    best, best_err = None, float('inf')
    for cnt in valid:
        try:
            ell = cv2.fitEllipseAMS(cnt)
            a, b = ell[1][0] / 2.0, ell[1][1] / 2.0
            if min(a, b) < 5:
                continue
            mean_r = (a + b) / 2.0
            err = abs(mean_r - expected_r)
            if err < best_err:
                best_err = err
                best = (int(ell[0][0]), int(ell[0][1]), mean_r)
        except Exception:
            continue

    if best is None or best_err > expected_r * tol:
        return None
    return best


def fit_ellipse_ring(roi, bw_px, bh_px):
    """task6 运行时：只返回圆心坐标"""
    result = _fit_ring_best(roi, bw_px, bh_px)
    return (result[0], result[1]) if result else None


def calibrate_ring6(roi, bw_px, bh_px):
    """startup 标定专用：返回 (cx, cy, r_px)"""
    result = _fit_ring_best(roi, bw_px, bh_px)
    if result is None:
        return None
    cx, cy, r_px = result
    return cx, cy, int(round(r_px))
