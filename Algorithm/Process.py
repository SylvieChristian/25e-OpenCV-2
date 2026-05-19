# -*- coding: utf-8 -*-
"""
Algorithm.Process — 黑色胶带目标检测流水线
============================================
detect_black_tape_roi() 主入口，内部分为 7 个可复用阶段。
"""
import cv2
import numpy as np
from Algorithm.Config import DetectorConfig
from Algorithm import utils


# ====================================================================
#  阶段 1: 大津二值化 + 形态学
# ====================================================================
def create_mask(frame, cfg: DetectorConfig):
    """灰度 + CLAHE(可选) + GaussianBlur + OTSU + 形态学 → 二值掩码"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 形态学清理
    if cfg.open_k > 0:
        ksize = cfg.open_k * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=cfg.open_i)
    if cfg.close_k > 0:
        ksize = cfg.close_k * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.close_i)

    return mask


# ====================================================================
#  阶段 1b: 全帧红色掩码（调试可视化）
# ====================================================================
def create_red_mask(frame, cfg: DetectorConfig):
    """生成全帧红色像素二值图，用于调试窗口显示"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, cfg.red_lower1, cfg.red_upper1),
        cv2.inRange(hsv, cfg.red_lower2, cfg.red_upper2),
    )
    return red_mask


# ====================================================================
#  阶段 2: 帧预处理（resize + 中心裁剪）
# ====================================================================
def preprocess(frame, cfg: DetectorConfig):
    """resize + 中心裁剪"""
    frame = cv2.resize(frame, (cfg.frameWidth, cfg.frameHeight))
    cx = int(cfg.frameWidth  * cfg.CROP_CX)
    cy = int(cfg.frameHeight * cfg.CROP_CY)
    cw = int(cfg.frameWidth  * cfg.CROP_W)
    ch = int(cfg.frameHeight * cfg.CROP_H)
    x0 = max(cx - cw // 2, 0)
    y0 = max(cy - ch // 2, 0)
    x1 = min(cx + cw // 2, cfg.frameWidth)
    y1 = min(cy + ch // 2, cfg.frameHeight)
    return frame[y0:y1, x0:x1]


# ====================================================================
#  阶段 3: 候选提取 — 轮廓遍历 + 红色像素验证
# ====================================================================
def extract_candidates(frame, mask, cfg: DetectorConfig, debug_img):
    """
    遍历 mask 轮廓，用红色像素数筛选候选 ROI。

    返回: (candidates, fallback_cnt, fallback_area)
      candidates    — 通过红色验证的候选列表
      fallback_cnt  — 最大面积轮廓（红色不够时兜底）
      fallback_area — 最大面积
    """
    fh, fw = frame.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    fallback_cnt = None
    fallback_area = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.area_min:
            continue

        # 跟踪面积最大的轮廓，作为兜底
        if area > fallback_area:
            fallback_area = area
            fallback_cnt = cnt

        x, y, w, h = cv2.boundingRect(cnt)
        margin = max(2, int(min(w, h) * 0.05))
        x0 = max(x + margin, 0)
        y0 = max(y + margin, 0)
        x1 = min(x + w - margin, fw)
        y1 = min(y + h - margin, fh)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        # 1. 红色像素验证
        red_count = utils.validate_red(
            roi,
            cfg.red_lower1, cfg.red_upper1,
            cfg.red_lower2, cfg.red_upper2,
        )

        # 红色不够 → 灰色虚线框标记
        if red_count < cfg.red_pixel_min:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (80, 80, 80), 1)
            continue

        candidates.append({
            "roi": roi, "cnt": cnt,
            "rect": (x0, y0, x1 - x0, y1 - y0),
            "red_count": red_count,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "bx": x,  "by": y,  "bw": w,  "bh": h,
        })

    return candidates, fallback_cnt, fallback_area


# ====================================================================
#  阶段 4: 最优候选选择（按红色像素密度）
# ====================================================================
def select_best_candidate(candidates):
    """按红色像素密度（red_count / 包围盒面积）选最优"""
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["red_count"] / (c["bw"] * c["bh"]))


# ====================================================================
#  阶段 5: 三级中心定位
# ====================================================================
def locate_center(best, frame_h, frame_w, cfg: DetectorConfig):
    """
    三级回退定位靶心：
      Level 1 — DIAG: 四边形对角线交点（仅未贴边时可用）
      Level 2 — HOUGH / ELLI: 霍夫圆 或 椭圆拟合
      Level 3 — CENT: 矩心兜底

    返回: (center, center_type, circle_count, circles, corners)
    """
    x0, y0 = best["x0"], best["y0"]
    bx, by, bw, bh = best["bx"], best["by"], best["bw"], best["bh"]

    center = center_type = corners = circles = gsmall = None
    circle_count = 0

    # 判断轮廓是否贴边（被裁切）
    clipped = utils.is_clipped(bx, by, bw, bh, frame_h, frame_w, cfg.edge_margin)

    # ── Level 1: DIAG ── 对角线交点（不贴边 + 找到4角）
    if not clipped:
        corners = utils.find_4corners(best["cnt"])
        if corners is not None and utils.is_quad_valid(corners):
            center = utils.diagonal_center(corners)
            if center is not None:
                center_type = "DIAG"

    # ── Level 2: 霍夫圆 / 椭圆 ──
    if center is None:
        if cfg.task_mode == "ellipse":
            lc = utils.fit_ellipse_ring(best["roi"], bw, bh)
            if lc is not None:
                center = (lc[0] + x0, lc[1] + y0)
                center_type = "ELLI"
        else:
            circle_count, circles, gsmall = utils.run_hough(
                best["roi"],
                param2=0.6 if clipped else None,
                min_r_rat=cfg.hough_min_r_rat,
                max_r_rat=cfg.hough_max_r_rat,
                min_dist_rat=cfg.hough_min_dist_rat,
            )
            if circles is not None:
                # 按半径加权平均圆心
                cx_h, cy_h = utils.weighted_center(circles, offset=(x0, y0))
                if cx_h is not None:
                    center = (cx_h, cy_h)
                    center_type = "HOUGH"

    # ── Level 3: CENT ── 矩心兜底
    if center is None:
        M = cv2.moments(best["cnt"])
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            center_type = "CENT"

    return center, center_type, circle_count, circles, corners, gsmall


# ====================================================================
#  阶段 6: 兜底 ROI — 无红色候选时取最大轮廓
# ====================================================================
def build_fallback_roi(frame, fallback_cnt):
    """无红色候选时，取面积最大轮廓，矩心作为 center"""
    if fallback_cnt is None:
        return None

    fh, fw = frame.shape[:2]
    x, y, w, h = cv2.boundingRect(fallback_cnt)
    margin = max(2, int(min(w, h) * 0.05))
    x0 = max(x + margin, 0)
    y0 = max(y + margin, 0)
    x1 = min(x + w - margin, fw)
    y1 = min(y + h - margin, fh)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    fb_center = None
    M = cv2.moments(fallback_cnt)
    if M["m00"] > 0:
        fb_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    return {
        "roi": roi, "cnt": fallback_cnt,
        "rect": (x0, y0, x1 - x0, y1 - y0),
        "red_count": 0, "circle_count": 0, "circles": None,
        "center": fb_center, "center_type": "CENT",
        "corners": None, "warped": None,
        "bx": x, "by": y, "bw": w, "bh": h,
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
    }


# ====================================================================
#  阶段 7: 调试绘制 — 轮廓、包围盒、角点、圆心、类型标签
# ====================================================================
def draw_roi_debug(debug_img, best, center, center_type, circles, corners, task_mode):
    """在 debug_img 上绘制轮廓、包围盒、角点、圆心、类型标签"""
    x0, y0, x1, y1 = best["x0"], best["y0"], best["x1"], best["y1"]
    bx, by, bw, bh = best["bx"], best["by"], best["bw"], best["bh"]

    # 轮廓、包围盒
    cv2.drawContours(debug_img, [best["cnt"]], -1, (255, 255, 0), 2)
    cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (80, 80, 255), 1)
    cv2.putText(debug_img,
                f"area={int(cv2.contourArea(best['cnt']))} "
                f"ROI={x1 - x0}x{y1 - y0} red={best['red_count']} mode={task_mode}",
                (x0, max(y0 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 角点
    if corners is not None:
        for pt in corners.astype(int):
            cv2.circle(debug_img, tuple(pt), 5, (255, 255, 0), -1)

    # 中心十字 + 类型标签
    if center is not None:
        ccx, ccy = center
        cv2.line(debug_img, (ccx - 10, ccy - 10), (ccx + 10, ccy + 10), (0, 0, 255), 2)
        cv2.line(debug_img, (ccx + 10, ccy - 10), (ccx - 10, ccy + 10), (0, 0, 255), 2)
        cv2.circle(debug_img, (ccx, ccy), 3, (0, 0, 255), -1)
        cv2.putText(debug_img, center_type, (ccx + 12, ccy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # 霍夫圆
    if circles is not None:
        for (cx, cy, r) in circles:
            cv2.circle(debug_img, (x0 + cx, y0 + cy), r, (0, 255, 0), 1)
            cv2.circle(debug_img, (x0 + cx, y0 + cy), 3, (0, 255, 0), -1)


# ====================================================================
#  主入口：编排上述 7 个阶段
# ====================================================================
def detect_black_tape_roi(frame, cfg: DetectorConfig):
    """
    检测黑色胶带轮廓，裁出候选 ROI，红色验证后定位靶心。

    参数:
      frame — 预处理后的 BGR 图像
      cfg   — DetectorConfig 实例（阈值、模式等）

    返回: (rois, debug_img, mask)
    """
    debug_img = frame.copy()

    # 1. 大津二值化 + 形态学 → 二值掩码
    mask = create_mask(frame, cfg)

    # 2. 轮廓遍历 + 红色验证 → 候选列表
    candidates, fallback_cnt, _ = extract_candidates(frame, mask, cfg, debug_img)

    rois = []
    gsmall = None

    if candidates:
        # 3. 选最优候选（红色密度最高）
        best = select_best_candidate(candidates)
        fh, fw = frame.shape[:2]

        # 4. 三级定位靶心（DIAG → HOUGH/ELLI → CENT）
        center, center_type, circle_count, circles, corners, gsmall = locate_center(
            best, fh, fw, cfg
        )

        rois.append({
            "roi": best["roi"], "rect": best["rect"],
            "cnt": best["cnt"], "red_count": best["red_count"],
            "circle_count": circle_count, "circles": circles,
            "center": center, "center_type": center_type,
            "corners": corners, "warped": None,
        })

        # 5. 绘制调试信息
        draw_roi_debug(debug_img, best, center, center_type, circles, corners,
                       cfg.task_mode)

    elif fallback_cnt is not None:
        # 无红色候选 → 兜底：取面积最大轮廓
        fb = build_fallback_roi(frame, fallback_cnt)
        if fb is not None:
            rois.append(fb)
            cv2.drawContours(debug_img, [fallback_cnt], -1, (0, 100, 255), 2)
            cv2.rectangle(debug_img,
                          (fb["x0"], fb["y0"]),
                          (fb["x1"], fb["y1"]), (0, 100, 255), 2)
            cv2.putText(debug_img, "disappeared",
                        (fb["x0"], max(fb["y0"] - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

    if not rois:
        cv2.putText(debug_img, "未检测到黑色胶带", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return rois, debug_img, mask, gsmall
