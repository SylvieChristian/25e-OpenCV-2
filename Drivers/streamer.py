import cv2
from Algorithm.Config import DetectorConfig


SOURCE      = "csi"
USB_DEVICE  = "/dev/video1"
VIDEO_PATH  = "/home/sylvie/code/Resources/25电赛e视频.mp4"


# ==================== 2. CSI摄像头GStreamer管道 ====================
def gstreamer_pipeline(
    # ========== 采集分辨率 ============
    # 3280x2464@21fps / 3280x1848@28fps / 1920x1080@30fps / 1640x1232@30fps / 1280x720@60fps
    capture_width  = 3280,          # 改为1280，配合720p 60fps更流畅
    capture_height = 1848,           # 改为720
    # ========== 显示/处理分辨率（可以任意缩放，由nvvidconv处理）==========
    display_width  = DetectorConfig.frameWidth,    # 输出给OpenCV的宽度，越小越省算力
    display_height = DetectorConfig.frameHeight,   # 输出给OpenCV的高度
    # ========== 帧率 ==========
    framerate = 28,                 # 1280x720支持60fps，比1080p@30fps更流畅
    # ========== 画面翻转 ==========
    flip_method = 2,                # 0=不翻转 1=顺时针90 2=180 3=逆时针90
                                    # 摄像头倒装时用2
    # ========== 防频闪（必须设置）==========
    aeantibanding = 1,              # 1=50Hz(国内日光灯) 2=60Hz 3=自动
                                    # 国内室内场景统一用1，防止mask出现横向条纹

    # ========== 自动曝光控制 ==========
    aelock = True,                 # True=锁定曝光（比赛时锁死） False=自动（调试时用）
    exposuretimerange = "31872000 31872000",
                                    # 曝光时间范围，单位纳秒(ns)
                                    # 下限最小13000（传感器硬件限制，不能更低）
                                    # 上限最大683709000
                                    # 两端相同=锁死，如"13000 13000"（最暗）
                                    # 调试建议先用"13000 683709000"全开让AE自动
                                    # 看画面稳定后记录Mean V，再缩小范围锁死

    gainrange = "1.0 1.0",          # 模拟增益，两端相同=锁死
                                    # 增益越大越亮但噪点越多，优先调曝光时间
                                    # 范围min=1.0 max=10.625（传感器硬件上限）

    # ========== 白平衡控制 ==========
    awblock = True,                # True=锁定白平衡 False=自动
                                    # 比赛时锁定，防止场景变化导致HSV漂移
    wbmode  = 1,                    # 白平衡模式（awblock=False时生效）
                                    # 0=自动 1=白炽灯 2=日光灯 3=日光 4=阴天 5=阴影
    ispdigitalgainrange = "1 1",    # ISP数字增益，一般不动保持1 1

    # ========== 色彩饱和度 ==========
    saturation = 2.0,               # 饱和度，范围0.0~2.0，默认1.0

    # ========== 锐度 ==========
    tnr_mode     = 1,               # 时域降噪 0=关 1=保守 2=激进
    tnr_strength = 0.5,             # 降噪强度 0.0~1.0
    ee_mode      = 1,               # 边缘增强 0=关 1=开
    ee_strength  = 0.5,             # 边缘增强强度 0.0~1.0
):
    return (
        f"nvarguscamerasrc "
        # 曝光参数
        f"aelock={str(aelock).lower()} "
        f"exposuretimerange=\"{exposuretimerange}\" "
        f"gainrange=\"{gainrange}\" "
        f"ispdigitalgainrange=\"{ispdigitalgainrange}\" "
        # 白平衡参数
        f"awblock={str(awblock).lower()} "
        f"wbmode={wbmode} "
        # 防频闪
        f"aeantibanding={aeantibanding} "
        # 色彩和降噪
        f"saturation={saturation} "
        f"tnr-mode={tnr_mode} "
        f"tnr-strength={tnr_strength} "
        f"ee-mode={ee_mode} "
        f"ee-strength={ee_strength} ! "
        # 分辨率和格式
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, "
        f"width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink sync=false drop=true max-buffers=1"
    )

# ==================== 3. USB摄像头输入流 ====================
def create_usb_stream(
    device_id=1,                    # 设备ID: 0=/dev/video0, 1=/dev/video1
    #1920×1080@90fps
    #1280×720@160fps
    # 640×480@160fps
    width=1280,                     # 宽度
    height=720,                     # 高度
    fps=160,                        # 帧率
    fourcc='MJPG',                  # 编码格式: 'MJPG', 'YUYV', 'H264'
    buffer_size=1,                  # 缓冲区大小（减少延迟）
    # 图像参数（如果摄像头支持）
    brightness=128,                 # 亮度 0-255
    contrast=128,                   # 对比度 0-255
    saturation=128,                 # 饱和度 0-255
    hue=0,                          # 色调 -180~180
    gamma=100,                      # Gamma 100=正常
    sharpness=128,                  # 锐度 0-255
    exposure_auto=True,             # 自动曝光
    white_balance_auto=True,        # 自动白平衡
):
    
    import cv2

    print(f"正在打开USB摄像头 /dev/video{device_id}...")
    print(f"配置: {width}x{height} @ {fps}fps ({fourcc})")

    # 打开摄像头
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头设备 /dev/video{device_id}")
        return None

    # 设置FOURCC编码格式
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 设置帧率
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 设置缓冲区大小（减少延迟）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    # 设置图像参数（如果摄像头支持）
    try:
        # 自动控制
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if exposure_auto else 1)  # 3=自动, 1=手动
        cap.set(cv2.CAP_PROP_AUTO_WB, 1 if white_balance_auto else 0)

        # 手动参数
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
        cap.set(cv2.CAP_PROP_HUE, hue)
        cap.set(cv2.CAP_PROP_GAMMA, gamma)
        cap.set(cv2.CAP_PROP_SHARPNESS, sharpness)
    except Exception as e:
        print(f"警告: 部分摄像头参数设置失败: {e}")

    # 验证实际参数
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print(f"USB摄像头实际参数:")
    print(f"  分辨率: {actual_width}x{actual_height}")
    print(f"  帧率: {actual_fps:.1f}fps")
    print(f"  格式: {actual_fourcc_str}")

    return cap

def stream_creat():
    if SOURCE == "csi":
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    elif SOURCE == "usb":
        cap = cv2.VideoCapture(device_id=1)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        
    return cap