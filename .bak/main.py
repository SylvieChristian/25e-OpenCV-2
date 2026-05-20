#系统库
import cv2,time
import threading
import numpy as np

#自定义库
import Drivers.streamer  as  streamer
from Algorithm.Config import DetectorConfig
import Algorithm.Process as  process
from Drivers.SSD1306 import SSD1306_IIC
from Drivers.OLED_Menu import Menu

#全局变量
fps_cnt, fps_t, fps_val = 0, time.time(),0.0

#主函数初始化错误处理
def detect(cap):
    if not cap.isOpened():
        print("[错误] 初始化失败")
        cap.release()
        return False

    ret,_ =cap.read()
    if not ret:
        print("[错误] 无法读取视频源")
        cap.release()
        return False
    return True

#窗口显示
def ShowWin(debug_img, red_mask,mask,gsmall,cfg: DetectorConfig):
    cv2.imshow("ROI Debug", debug_img)
    cv2.imshow("Red Filter", red_mask)
    # cv2.imshow("gsmall", gsmall)
    # cv2.imshow("mask",mask)

#计算fps+显示
def fps(frame):
    global fps_cnt, fps_t, fps_val

    fps_cnt += 1
    now = time.time()
    if now - fps_t >= 1.0:
        fps_val = fps_cnt / (now - fps_t)
        fps_cnt, fps_t = 0, now
    cv2.putText(frame, f"FPS:{fps_val:.1f}", (0,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

#计算平均亮度+显示
def mean_v(frame):
    mean_v = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))[2]
    cv2.putText(frame, f"A_V:{mean_v:.1f}", (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

# ---- Trackbar ----
def _set_red_lo(cfg, idx, val):
    """红色下限数组：同步修改 lower1/lower2 的同一通道"""
    cfg.red_lower1[idx] = val
    cfg.red_lower2[idx] = val
def _set_red_hi(cfg, idx, val):
    """红色上限数组：同步修改 upper1/upper2 的同一通道"""
    cfg.red_upper1[idx] = val
    cfg.red_upper2[idx] = val

def Trackbars(cfg):
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 460, 100)
    cv2.namedWindow("Controls2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls2", 460, 100)

    # 形态学
    cv2.createTrackbar("Open  K", "Controls", cfg.open_k,  10, lambda v: setattr(cfg, 'open_k', v))
    cv2.createTrackbar("Open  I", "Controls", cfg.open_i,   5, lambda v: setattr(cfg, 'open_i', v))
    cv2.createTrackbar("Close K", "Controls", cfg.close_k, 10, lambda v: setattr(cfg, 'close_k', v))
    cv2.createTrackbar("Close I", "Controls", cfg.close_i,  5, lambda v: setattr(cfg, 'close_i', v))

    # 筛选
    cv2.createTrackbar("Area Min", "Controls", cfg.area_min, 800, lambda v: setattr(cfg, 'area_min', v))
    cv2.createTrackbar("CLAHE",    "Controls", int(cfg.use_clahe), 1, lambda v: setattr(cfg, 'use_clahe', bool(v)))

    # 红色阈值：slider 改数组单元素（原地修改，不建新数组）
    cv2.createTrackbar("Red S Min", "Controls2", int(cfg.red_lower1[1]), 255, lambda v: _set_red_lo(cfg, 1, v))
    cv2.createTrackbar("Red V Min", "Controls2", int(cfg.red_lower1[2]), 255, lambda v: _set_red_lo(cfg, 2, v))
    cv2.createTrackbar("Red S Max", "Controls2", int(cfg.red_upper1[1]), 255, lambda v: _set_red_hi(cfg, 1, v))
    cv2.createTrackbar("Red V Max", "Controls2", int(cfg.red_upper1[2]), 255, lambda v: _set_red_hi(cfg, 2, v))
    cv2.createTrackbar("RedPxMin",  "Controls2", cfg.red_pixel_min, 800, lambda v: setattr(cfg, 'red_pixel_min', v))

    # 霍夫参数：int slider → float 属性
    cv2.createTrackbar("Hough P2",     "Controls2", int(cfg.hough_param2 * 100),     100, lambda v: setattr(cfg, 'hough_param2', v / 100.0))
    cv2.createTrackbar("Min R x100",   "Controls2", int(cfg.hough_min_r_rat * 100),   30, lambda v: setattr(cfg, 'hough_min_r_rat', v / 100.0))
    cv2.createTrackbar("Max R x100",   "Controls2", int(cfg.hough_max_r_rat * 100),   80, lambda v: setattr(cfg, 'hough_max_r_rat', v / 100.0))
    cv2.createTrackbar("MinDist x100", "Controls2", int(cfg.hough_min_dist_rat * 100), 50, lambda v: setattr(cfg, 'hough_min_dist_rat', v / 100.0))


#菜单线程 — 独立线程跑
#格式： ''name'' 显示名称，''get'' 读取当前值的函数
#       ''set'' 修改值的函数，''choices'' 可选项列表
#       ''range'' (lo, hi, step) 数值范围和步长
def menu_loop(cfg):
    oled = SSD1306_IIC(cfg.i2cDeviceIndex, with_keys=True)
    menu = Menu(oled, [
        {'name': 'FPS ', 'get': lambda: fps_val},
        {'name': 'Task', 'get': lambda: cfg.task_mode,
                         'set': lambda v: setattr(cfg, 'task_mode', v),
                         'choices': ['normal', 'ellipse']},
        {'name': 'Area', 'get': lambda: cfg.area_min,
                         'set': lambda v: setattr(cfg, 'area_min', v),
                         'range': (0, 8000, 100)},
        {'name': 'P2',   'get': lambda: cfg.hough_param2,
                         'set': lambda v: setattr(cfg, 'hough_param2', round(v, 2)),
                         'range': (0.0, 1.0, 0.05)},
        {'name': 'CLAHE','get': lambda: cfg.use_clahe,
                         'set': lambda v: setattr(cfg, 'use_clahe', v),
                         'choices': [False, True]},
        {'name': '<RedPixel>', 'get': lambda: cfg.red_pixel_min,
                         'set': lambda v: setattr(cfg, 'red_pixel_min', v),
                         'range': (0, 800, 10)
                         },
    ])
    try:
        while True:
            menu.step(timeout_ms=80)
    finally:
        oled.close()


#标准亮度197
def main():
    cap= streamer.stream_creat()        #获取摄像头数据流

    if not detect(cap):
        return
    try:
        cfg = DetectorConfig()
        Trackbars(cfg)
        threading.Thread(target=menu_loop, args=(cfg,), daemon=True).start()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流中断或结束")
                break

            #原始数据裁切
            frame = process.preprocess(frame, cfg)

            #图像处理 → 返回 (rois, debug_img, mask)
            _, debug_img, mask , gsmall = process.detect_black_tape_roi(frame, cfg)

            #全帧红色掩码（调试可视化）
            red_mask = process.create_red_mask(frame, cfg)

            #debug信息放置
            mean_v  (debug_img)
            fps     (debug_img)

            #窗口显示
            ShowWin(debug_img, red_mask, mask, gsmall, cfg)

            #调试key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                # ESC
                break
            elif key == ord('q'):        # 按 q 也可以退出
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
