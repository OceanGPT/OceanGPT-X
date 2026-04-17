# 海洋图像检索 API

一个基于 FastAPI 的海洋图像识别服务，支持：

- 基于 BioCLIP 特征 + FAISS 的近邻检索（相似度 ≥ 0.90 直接返回）
- 多模型融合与交叉验证识别
- 声呐图像目标分类（15 类）与生物图像识别（鱼类、珊瑚种级识别）

## 架构

```
输入图像
  |
  v
[1] FAISS 检索（BioCLIP 特征，相似度 ≥ 0.90 → 直接返回）
  |  未命中
  v
[2] Router 二分类器（声呐 / 生物，YOLOv11-cls）
  |
  +-- 声呐 → [3a] 声呐分类器（15 类，YOLOv5）→ fusion
  |
  +-- 生物 → [3b] 鱼类/珊瑚二分类（YOLOv5）
                  |
                  +-- 鱼类 → 鱼类检测器（YOLOv5）---+
                  +-- 珊瑚 → 珊瑚检测器（YOLOv5）---+→ fusion（+ OceanCLIP）
  v
[4] Fusion：将 OceanCLIP 物种匹配结果与检测器交叉验证，
            回退到最高置信度候选
```

## 快速开始

### 1. 安装依赖

```bash
conda env create -f environment.yml
conda activate marine-api
```

### 2. 克隆 YOLOv5 源码

鱼类、珊瑚、声呐模型均为 YOLOv5 格式，需要 YOLOv5 源码来加载：

```bash
git clone https://github.com/ultralytics/yolov5 /path/to/yolov5
```

请将 `/path/to/yolov5` 替换为你自己的路径，后续需要在 `.env` 中配置。

### 3. 下载模型权重与数据文件

所有模型权重和索引数据均托管在 Hugging Face 上：
**[zhemaxiya/marine-image-api-models](https://huggingface.co/zhemaxiya/marine-image-api-models)**

运行以下命令一键下载：

```bash
python scripts/download_assets.py
```

该脚本会下载：
- 7 个模型权重（Router、声呐分类器、鱼类/珊瑚二分类、鱼类检测器、珊瑚检测器、OceanCLIP 微调权重 + 术语表）
- BioCLIP 基础模型（用于 FAISS 特征编码）
- FAISS 检索索引
- 元数据文件

自定义下载目录：

```bash
python scripts/download_assets.py --download-dir ./my-models
```

运行完成后，脚本会自动输出所有需要配置的环境变量 export 命令。

### 4. 配置环境变量（可选）

所有路径默认指向 `downloaded_assets/` 目录（下载脚本自动创建），**无需手动配置即可启动**。

仅在以下情况需要设置环境变量：

- YOLOv5 克隆到了非默认路径：
  ```bash
  export YOLOV5_DIR=/path/to/yolov5
  ```
- 调整推理参数：
  ```bash
  export THRESHOLD=0.85
  export TOPK=10
  ```

### 5. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

开发模式（支持热更新）：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Streamlit 演示界面

启动交互式 Web 演示：

```bash
streamlit run streamlit/demo.py
```

## API 接口

### 健康检查

```
GET /health
```

### 预测接口

```
POST /predict
```

| 字段 | 类型  | 说明     |
|------|-------|----------|
| file | image | 待识别的图片 |

**使用示例：**

```bash
curl -X POST http://localhost:8000/predict -F "file=@test/soner_cube.png"
```

或打开 `http://localhost:8000/docs` 查看交互式 API 文档。

## 配置说明

关键环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `THRESHOLD` | `0.90` | FAISS 检索相似度阈值 |
| `ROUTER_THRESHOLD` | `0.5` | 声呐分类概率阈值，高于此值判定为声呐图 |
| `USE_OCEANCLIP` | `true` | 是否启用 OceanCLIP 物种级识别 |
| `TOPK` | `5` | FAISS 返回的最邻近数量 |
| `DEVICE` | `cuda` | 计算设备（`cuda` 或 `cpu`） |

## 项目结构

```
app/
  api/          # FastAPI 路由（/health, /predict）
  core/         # 配置与全局状态
  services/     # 模型加载、检索、分类、融合等核心逻辑
  main.py       # 应用入口
scripts/
  download_assets.py  # 一键下载所有模型和数据文件
streamlit/
  demo.py       # Streamlit Web 演示界面
test/           # 测试样例图片
```

## 测试样例

`test/` 目录下提供 4 张测试图片：

- `test/coral_Acropora Cervicornis_1.png` — 珊瑚（Acropora cervicornis）
- `test/fish_Amphiprion_clarkii_62.png` — 鱼类（Amphiprion clarkii）
- `test/soner_cube.png` — 声呐（cube）
- `test/fish.png` — 域外风格鱼图（水族箱白背景）

## 说明

本仓库不包含模型权重和数据文件，请通过 `scripts/download_assets.py` 下载并配置 `.env` 中的路径。
