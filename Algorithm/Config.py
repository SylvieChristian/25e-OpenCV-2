# -*- coding: utf-8 -*-
"""
Algorithm.Config — 检测器配置
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DetectorConfig:
    """可调参数集合，运行时 trackbar 直接改实例属性"""

    # 红色阈值
    red_lower1: np.ndarray = field(default_factory=lambda: np.array([0, 95, 160]))
    red_upper1: np.ndarray = field(default_factory=lambda: np.array([10, 255, 255]))
    red_lower2: np.ndarray = field(default_factory=lambda: np.array([160, 95, 160]))
    red_upper2: np.ndarray = field(default_factory=lambda: np.array([180, 255, 255]))
    red_pixel_min: int = 5000

    # 形态学
    open_k:  int = 2
    open_i:  int = 1
    close_k: int = 2
    close_i: int = 2

    # 筛选 & 边缘
    area_min:    int = 4900
    edge_margin: int = 5
    use_clahe:   bool = False

    # 定位参数
    hough_param2: float = 0.35
    hough_min_r_rat: float = 0.04      #最小圆半径 (10px@250short)
    hough_max_r_rat: float = 0.7      #最大圆半径 (70px@250short)
    hough_min_dist_rat: float = 0.52   #圆心间距  (130px@250short)
    task_mode: str = "normal"  # "normal" | "ellipse"

    # 视频尺寸
    frameWidth:  int = int(3280 * 0.35)
    frameHeight: int = int(1848 * 0.35)

    # 预处理裁剪
    CROP_CX: float = 0.5
    CROP_CY: float = 0.5
    CROP_W:  float = 0.5
    CROP_H:  float = 0.5


# ====================================================================
#  模块级默认常量（供 utils.py 内部 fallback 用）
# ====================================================================
RED_LOWER1 = np.array([0, 95, 160])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 95, 160])
RED_UPPER2 = np.array([180, 255, 255])

RED_PIXEL_MIN = 5000

HOUGH_PARAM2    = 0.35
HOUGH_MIN_R_RAT = 0.04
HOUGH_MAX_R_RAT = 0.7
HOUGH_MIN_DIST_RAT = 0.52

WARP_SCALE = 1.5
