#系统库
import cv2,time
import numpy as np

#自定义库
import Drivers.streamer  as  streamer
from Algorithm.Config import DetectorConfig
import Algorithm.Process as  process
from Drivers.SSD1306 import SSD1306_IIC
from Drivers.OLED_Menu import Menu

#全局变量
fps_cnt, fps_t, fps_val = 0, time.time(), 0.0

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

#滑块调节窗口
def Trackbars(cfg):
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 460, 100)
    cv2.namedWindow("Controls2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls2", 460, 100)

    def nothing(x): pass

    # 形态学
    cv2.createTrackbar("Open  K", "Controls", cfg.open_k,  10, nothing)
    cv2.createTrackbar("Open  I", "Controls", cfg.open_i,   5, nothing)
    cv2.createTrackbar("Close K", "Controls", cfg.close_k, 10, nothing)
    cv2.createTrackbar("Close I", "Controls", cfg.close_i,  5, nothing)

    # 筛选
    cv2.createTrackbar("Area Min", "Controls", cfg.area_min, 800 , nothing)
    cv2.createTrackbar("CLAHE",    "Controls", int(cfg.use_clahe), 1, nothing)

    # 红色阈值 (H 范围固定，S/V 可调上下限)
    cv2.createTrackbar("Red S Min", "Controls2", int(cfg.red_lower1[1]), 255, nothing)
    cv2.createTrackbar("Red V Min", "Controls2", int(cfg.red_lower1[2]), 255, nothing)
    cv2.createTrackbar("Red S Max", "Controls2", int(cfg.red_upper1[1]), 255, nothing)
    cv2.createTrackbar("Red V Max", "Controls2", int(cfg.red_upper1[2]), 255, nothing)
    cv2.createTrackbar("RedPxMin",  "Controls2", cfg.red_pixel_min,     800, nothing)

    # 霍夫圆检测参数
    cv2.createTrackbar("Hough P2", "Controls2",     int(cfg.hough_param2 * 100), 100, nothing)
    cv2.createTrackbar("Min R x100",   "Controls2", int(cfg.hough_min_r_rat * 100),   30, nothing)
    cv2.createTrackbar("Max R x100",   "Controls2", int(cfg.hough_max_r_rat * 100),   80, nothing)
    cv2.createTrackbar("MinDist x100", "Controls2", int(cfg.hough_min_dist_rat * 100), 50, nothing)

#读取滑块值同步到cfg
def update_trackbars(cfg):
    cfg.open_k  = cv2.getTrackbarPos("Open  K", "Controls")
    cfg.open_i  = cv2.getTrackbarPos("Open  I", "Controls")
    cfg.close_k = cv2.getTrackbarPos("Close K", "Controls")
    cfg.close_i = cv2.getTrackbarPos("Close I", "Controls")
    cfg.area_min    = cv2.getTrackbarPos("Area Min", "Controls")
    cfg.use_clahe   = cv2.getTrackbarPos("CLAHE",    "Controls") == 1

    red_s_lo = cv2.getTrackbarPos("Red S Min", "Controls2")
    red_v_lo = cv2.getTrackbarPos("Red V Min", "Controls2")
    red_s_hi = cv2.getTrackbarPos("Red S Max", "Controls2")
    red_v_hi = cv2.getTrackbarPos("Red V Max", "Controls2")
    cfg.red_lower1 = np.array([0,   red_s_lo, red_v_lo])
    cfg.red_upper1 = np.array([10,  red_s_hi, red_v_hi])
    cfg.red_lower2 = np.array([160, red_s_lo, red_v_lo])
    cfg.red_upper2 = np.array([180, red_s_hi, red_v_hi])
    cfg.red_pixel_min = cv2.getTrackbarPos("RedPxMin", "Controls2")

    cfg.hough_param2 = cv2.getTrackbarPos("Hough P2", "Controls2") / 100.0
    cfg.hough_min_r_rat   = cv2.getTrackbarPos("Min R x100",   "Controls2") / 100.0
    cfg.hough_max_r_rat   = cv2.getTrackbarPos("Max R x100",   "Controls2") / 100.0
    cfg.hough_min_dist_rat = cv2.getTrackbarPos("MinDist x100", "Controls2") / 100.0


#标准亮度197
def main():
    cap= streamer.stream_creat()        #获取摄像头数据流

    if not detect(cap):
        return
    try:
        cfg = DetectorConfig()
        Trackbars(cfg)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流中断或结束")
                break

            update_trackbars(cfg)

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
