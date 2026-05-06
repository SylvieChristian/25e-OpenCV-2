# 电赛25E 视觉定位

## 环境

Python 3.8.10 + OpenCV 4.5.3 + NumPy 1.21.2 + CUDA 11.4

## 硬件
Jetson Orin Nano + imx 219 CSI 摄像头

两个 42步进云台（无编码器）+mspm0g3507+jy901+drv8825


## 功能

黑色胶带靶心检测。OTSU 二值化 + 红色 HSV 验证 + 三级定位（DIAG/HOUGH/CENT）。

## 结构

```
认真学习/
├── main.py              # 主循环、显示、滑块
├── Algorithm/
│   ├── Config.py        # 参数集
│   ├── Process.py       # 检测、定位、校准
│   └── utils.py         # 被调用 几何、霍夫、椭圆工具
└── Drivers/
    └── streamer.py      # 视频流配置
```




## 三级定位策略

1. DIAG —   四边形对角线交点（未被裁切）
2. HOUGH —  霍夫圆加权中心  （被裁切）
3. CENT —   矩心兜底        （没招了最后的默认值）

