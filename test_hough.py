# -*- coding: utf-8 -*-
"""霍夫圆参数测试 — 单独调试 HoughCircles"""
import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/sylvie/code/openCV/认真学习')

import time
import Drivers.streamer as streamer
from Algorithm.Config import DetectorConfig
import Algorithm.Process as process
from Algorithm import utils

# 全局变量
fps_cnt, fps_t, fps_val = 0, time.time(), 0.0


def nothing(x):
    pass


def show_fps(img):
    global fps_cnt, fps_t, fps_val
    fps_cnt += 1
    now = time.time()
    if now - fps_t >= 1.0:
        fps_val = fps_cnt / (now - fps_t)
        fps_cnt, fps_t = 0, now
    cv2.putText(img, f"FPS:{fps_val:.1f}", (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def show_mean_v(img):
    mean_v = cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
    cv2.putText(img, f"A_V:{mean_v:.1f}", (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def main():
    cap = streamer.stream_creat()
    if not cap.isOpened():
        print("[错误] 摄像头打开失败")
        return

    cfg = DetectorConfig()

    # ── 创建窗口 ──
    cv2.namedWindow("Hough Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hough Test", 960, 540)

    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Edges", 480, 360)

    cv2.namedWindow("Hough Sliders", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hough Sliders", 500, 300)

    # ── 霍夫滑块（下限均为 1，防止非正数报错）──
    cv2.createTrackbar("param2 x100", "Hough Sliders", 35, 100, nothing)   # 0.01~1.0
    cv2.createTrackbar("dp x10",      "Hough Sliders", 15, 30, nothing)    # 0.1~3.0
    cv2.createTrackbar("param1",      "Hough Sliders", 40, 300, nothing)   # 1~300
    cv2.createTrackbar("Min R",       "Hough Sliders", 10, 200, nothing)   # 1~200
    cv2.createTrackbar("Max R",       "Hough Sliders", 70, 300, nothing)   # 1~300
    cv2.createTrackbar("MinDist",     "Hough Sliders", 130,200, nothing)   # 1~200
    cv2.createTrackbar("Blur K x2+1", "Hough Sliders", 2,  10,  nothing)   # 1~11 odd
    cv2.createTrackbar("Scale x10",   "Hough Sliders", 8,  10,  nothing)   # 0.1~1.0

    print("[测试] ESC/q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频流中断")
            break

        # 预处理（和主流程一致）
        frame = process.preprocess(frame, cfg)

        # 生成掩码 + 提取候选
        mask = process.create_mask(frame, cfg)
        debug_img = frame.copy()
        candidates, fallback_cnt, _ = process.extract_candidates(frame, mask, cfg, debug_img)

        # 确定 ROI 来源
        if candidates:
            best = process.select_best_candidate(candidates)
            roi = best["roi"]
            x0, y0 = best["x0"], best["y0"]
            bx, by, bw, bh = best["bx"], best["by"], best["bw"], best["bh"]
            source = "candidate"
        elif fallback_cnt is not None:
            fb = process.build_fallback_roi(frame, fallback_cnt)
            if fb is None:
                cv2.imshow("Hough Test", debug_img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')): break
                continue
            roi = fb["roi"]
            x0, y0 = fb["x0"], fb["y0"]
            bx, by, bw, bh = fb["bx"], fb["by"], fb["bw"], fb["bh"]
            source = "fallback"
        else:
            cv2.putText(debug_img, "无目标", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Hough Test", debug_img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break
            continue

        if roi is None or roi.size == 0:
            continue

        # ── 读取滑块（全部钳位到安全范围）──
        param2    = max(0.01, cv2.getTrackbarPos("param2 x100", "Hough Sliders") / 100.0)
        dp        = max(0.1,  cv2.getTrackbarPos("dp x10",      "Hough Sliders") / 10.0)
        param1    = max(1,    cv2.getTrackbarPos("param1",      "Hough Sliders"))
        min_r     = max(1,    cv2.getTrackbarPos("Min R",       "Hough Sliders"))
        max_r     = max(min_r + 1, cv2.getTrackbarPos("Max R",  "Hough Sliders"))
        min_dist  = max(1,    cv2.getTrackbarPos("MinDist",     "Hough Sliders"))
        blur_k    = max(3,    cv2.getTrackbarPos("Blur K x2+1", "Hough Sliders") * 2 + 1)
        scale     = max(0.1,  cv2.getTrackbarPos("Scale x10",   "Hough Sliders") / 10.0)

        # ── 霍夫圆检测 ──
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        gsmall = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))))
        gsmall = cv2.GaussianBlur(gsmall, (blur_k, blur_k), 0)

        # Canny 边缘（可视化 param1 的效果）
        edges = cv2.Canny(gsmall, param1 / 2, param1)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(
            gsmall, cv2.HOUGH_GRADIENT_ALT, dp=dp,
            minDist=min_dist, param1=param1, param2=param2,
            minRadius=min_r, maxRadius=max_r,
        )

        # ── 绘制 ──
        roi_disp = roi.copy()
        cv2.rectangle(roi_disp, (0, 0), (roi_disp.shape[1]-1, roi_disp.shape[0]-1), (0, 255, 255), 2)

        circle_count = 0
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            circles[:, :2] = (circles[:, :2] / scale).astype(int)
            circles[:, 2]  = (circles[:, 2]  / scale).astype(int)
            circle_count = len(circles)

            for (cx, cy, r) in circles:
                cv2.circle(roi_disp, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(roi_disp, (cx, cy), 2, (0, 0, 255), -1)

            # 加权圆心
            cx_h, cy_h = utils.weighted_center(circles)
            if cx_h is not None:
                cv2.drawMarker(roi_disp, (cx_h, cy_h), (255, 0, 0),
                               cv2.MARKER_CROSS, 20, 2)

        # 信息叠加
        cv2.putText(roi_disp, f"circles:{circle_count} src:{source}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(roi_disp, f"p2:{param2:.2f} dp:{dp:.1f} p1:{param1}",
                    (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(roi_disp, f"r:[{min_r},{max_r}] dist:{min_dist} blur:{blur_k}",
                    (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(roi_disp, f"ROI:{w}x{h} scale:{scale:.1f}",
                    (5, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 在 debug_img 上也画包围盒 + FPS/亮度
        cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
        cv2.putText(debug_img, f"ROI -> Hough Test", (bx, max(by - 8, 12)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        show_fps(debug_img)
        show_mean_v(debug_img)

        cv2.imshow("Hough Test", roi_disp)
        cv2.imshow("Overview", debug_img)
        cv2.imshow("Edges", edges_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
