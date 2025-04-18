import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import sys
import os
import subprocess
import pkg_resources
import argparse
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import ColorMode
import copy
import tempfile
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmdet.utils import register_all_modules as register_det_modules
from mmpose.utils import register_all_modules as register_pose_modules
from mmpose.structures import merge_data_samples
from mmdet.structures import DetDataSample


# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvironmentManager:
    """环境管理类，负责依赖安装和环境初始化"""
    
    BASIC_DEPENDENCIES = {
        'numpy', 
        'opencv-python', 
        'pillow', 
        'photoshop-python-api',
        'torch',
        'torchvision'
    }
    
    DWPOSE_DEPENDENCIES = {
        'mmengine>=0.7.1',
        'mmcv>=2.0.0',
        'mmdet>=3.0.0',
        'mmpose>=1.1.0'
    }
    
    MODEL_FILES = {
        'rtmdet': {
            'name': 'rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth',
            'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
        },
        'rtmpose': {
            'name': 'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
            'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
        }
    }

    def __init__(self, dwpose_dir: str = r"D:\drafts\DWPose"):
        self.dwpose_dir = dwpose_dir
        self.weights_dir = os.path.join(dwpose_dir, 'weights')
        
    def setup(self) -> Dict[str, str]:
        """完整的环境设置流程"""
        try:
            self._check_and_install_dependencies()
            self._ensure_directories()
            self._download_model_files()
            self._register_modules()
            return self._get_model_paths()
        except Exception as e:
            logger.error(f"环境设置失败: {str(e)}")
            raise

    def _check_and_install_dependencies(self) -> None:
        """检查并安装依赖"""
        logger.info("检查依赖包...")
        installed = {pkg.key for pkg in pkg_resources.working_set}
        
        # 检查基础依赖
        missing_basic = self.BASIC_DEPENDENCIES - installed
        if missing_basic:
            self._install_packages(missing_basic, "基础依赖")
            
        # 安装DWPose依赖
        self._install_packages(self.DWPOSE_DEPENDENCIES, "DWPose依赖")

    def _install_packages(self, packages: Set[str], package_type: str) -> None:
        """安装指定的包"""
        logger.info(f"开始安装{package_type}...")
        python = sys.executable
        
        for pkg in packages:
            logger.info(f"正在安装 {pkg}...")
            try:
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                result = subprocess.run(
                    [python, '-m', 'pip', 'install', pkg, '--progress-bar', 'on'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    check=True  # 这会在安装失败时抛出异常
                )
                logger.info(f"{pkg} 安装成功")
            except subprocess.CalledProcessError as e:
                logger.error(f"{pkg} 安装失败: {e.stderr}")
                raise

    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        os.makedirs(self.weights_dir, exist_ok=True)

    def _download_model_files(self) -> None:
        """下载模型文件"""
        for model_type, info in self.MODEL_FILES.items():
            model_path = os.path.join(self.weights_dir, info['name'])
            if not os.path.exists(model_path):
                logger.info(f"正在下载{model_type}模型...")
                try:
                    import urllib.request
                    urllib.request.urlretrieve(info['url'], model_path)
                    logger.info(f"{model_type}模型下载完成")
                except Exception as e:
                    logger.error(f"下载{model_type}模型失败: {str(e)}")
                    logger.info(f"请手动下载模型文件: {info['url']}")
                    logger.info(f"并将其放置在: {model_path}")
                    raise

    def _register_modules(self) -> None:
        """注册必要的模块"""
        register_det_modules()
        register_pose_modules()

    def _get_model_paths(self) -> Dict[str, str]:
        """获取模型路径配置"""
        return {
            'det_config': os.path.join(
                self.dwpose_dir,
                'mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
            ),
            'det_checkpoint': os.path.join(
                self.weights_dir,
                self.MODEL_FILES['rtmdet']['name']
            ),
            'pose_config': os.path.join(
                self.dwpose_dir,
                'mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
            ),
            'pose_checkpoint': os.path.join(
                self.weights_dir,
                self.MODEL_FILES['rtmpose']['name']
            )
        }

class Config:
    """基础配置类"""
    def __init__(self, dwpose_dir: str = r"D:\drafts\DWPose"):
        self.dwpose_dir = dwpose_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = 0.3
        self.num_keypoints = 17
        self.model_paths = self._init_model_paths()
        self._register_modules()

    def _init_model_paths(self) -> Dict[str, str]:
        """初始化模型路径"""
        weights_dir = os.path.join(self.dwpose_dir, 'weights')
        return {
            'det_config': os.path.join(
                self.dwpose_dir,
                'mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
            ),
            'det_checkpoint': os.path.join(
                weights_dir,
                'rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
            ),
            'pose_config': os.path.join(
                self.dwpose_dir,
                'mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
            ),
            'pose_checkpoint': os.path.join(
                weights_dir,
                'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
            )
        }

    def _register_modules(self) -> None:
        """注册必要的模块"""
        register_det_modules()
        register_pose_modules()

class PoseDetector:
    """姿态检测器类"""
    def __init__(self, config: Config):
        self.config = config
        self._init_models()
        
    def _init_models(self):
        """初始化模型"""
        try:
            # 1. 先初始化姿态估计模型
            register_pose_modules()  # 注册姿态模块
            import mmpose.datasets   # 导入姿态数据集
            
            self.pose_estimator = init_pose_estimator(
                self.config.model_paths['pose_config'],
                self.config.model_paths['pose_checkpoint'],
                device=self.config.device
            )
            logger.info("姿态估计模型加载成功")
            
            # 2. 然后初始化检测器模型
            register_det_modules()   # 注册检测模块
            import mmdet.datasets    # 导入检测数据集
            
            self.detector = init_detector(
                self.config.model_paths['det_config'],
                self.config.model_paths['det_checkpoint'],
                device=self.config.device
            )
            logger.info("检测器模型加载成功")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise

    def detect_pose(self, image: np.ndarray, image_type: str = "未知") -> Dict:
        """检测姿态关键点"""
        try:
            print(f"\n==== 开始检测{image_type}图像 ====")
            
            # 确保图像格式正确
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                print(f"{image_type}图像为灰度图像，已转换为BGR图像")
            
            # 1. 使用检测器前，注册检测器相关模块
            register_det_modules()  # 在调用检测器前注册检测模块
            import mmdet.datasets    # 确保检测相关的transform被导入
            
            # 使用检测器获取人体边界框
            det_result = inference_detector(self.detector, image)
            
            # 检查结果并处理边界情况
            if not det_result or not hasattr(det_result, 'pred_instances'):
                print(f"❌ {image_type}图像检测失败：未检测到预测实例")
                logger.warning(f"在{image_type}图像中未检测到预测实例")
                return self._empty_result((image.shape[0], image.shape[1]))
            
            if not hasattr(det_result.pred_instances, 'bboxes') or det_result.pred_instances.bboxes.shape[0] == 0:
                print(f"❌ {image_type}图像检测失败：未检测到人体边界框")
                logger.warning(f"在{image_type}图像中未检测到人体边界框")
                return self._empty_result((image.shape[0], image.shape[1]))
            
            # 获取预测实例中的边界框
            bboxes = det_result.pred_instances.bboxes.cpu().numpy()
            scores = det_result.pred_instances.scores.cpu().numpy()
            
            # 打印检测到的人体信息
            print(f"✅ 在{image_type}图像中检测到 {len(bboxes)} 个人体")
            for i, (bbox, score) in enumerate(zip(bboxes, scores)):
                print(f"  人体 #{i+1}: 位置=[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}], 置信度={score:.4f}")
            
            # 2. 使用姿态估计器前，注册姿态估计相关模块
            register_pose_modules()  # 在调用姿态估计器前注册姿态模块
            import mmpose.datasets   # 确保姿态相关的transform被导入
            
            # 使用姿态估计器进行推理
            pose_results = inference_topdown(self.pose_estimator, image, bboxes)
            
            # 如果没有姿态结果，返回空结果
            if not pose_results:
                print(f"❌ {image_type}图像姿态估计失败：未检测到姿态关键点")
                logger.warning(f"在{image_type}图像中未检测到姿态关键点")
                return self._empty_result((image.shape[0], image.shape[1]))
            
            # 安全地获取关键点信息
            try:
                pred_instance = pose_results[0].pred_instances
                
                # 提取关键点和分数
                keypoints = []
                keypoint_scores = []
                
                if hasattr(pred_instance, 'keypoints'):
                    if torch.is_tensor(pred_instance.keypoints):
                        keypoints = pred_instance.keypoints[0].cpu().numpy()
                    else:
                        keypoints = pred_instance.keypoints[0]
                        
                    if hasattr(pred_instance, 'keypoint_scores'):
                        if torch.is_tensor(pred_instance.keypoint_scores):
                            keypoint_scores = pred_instance.keypoint_scores[0].cpu().numpy()
                        else:
                            keypoint_scores = pred_instance.keypoint_scores[0]
                    else:
                        keypoint_scores = np.ones(len(keypoints))
                    
                    # 打印检测到的关键点信息
                    valid_keypoints = [i for i, score in enumerate(keypoint_scores) if score > self.config.confidence_threshold]
                    print(f"✅ 在{image_type}图像中检测到 {len(valid_keypoints)}/{len(keypoints)} 个关键点 (置信度>{self.config.confidence_threshold})")
                    
                    # 打印关键点名称和位置
                    keypoint_names = [
                        "鼻子", "颈部", "右肩", "右肘", "右腕",
                        "左肩", "左肘", "左腕", "右髋", "右膝",
                        "右踝", "左髋", "左膝", "左踝", "右眼",
                        "左眼", "右耳", "左耳"
                    ]
                    
                    for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
                        if i < len(keypoint_names) and score > self.config.confidence_threshold:
                            print(f"  关键点 #{i} {keypoint_names[i]}: 位置=[{int(kpt[0])},{int(kpt[1])}], 置信度={score:.4f}")
                    
                else:
                    print(f"❌ {image_type}图像姿态估计失败：未检测到关键点属性")
                    logger.warning(f"在{image_type}图像中未检测到关键点属性")
            except Exception as inner_e:
                print(f"❌ {image_type}图像姿态估计处理错误: {str(inner_e)}")
                logger.warning(f"处理关键点数据时出错: {str(inner_e)}")
                return self._empty_result((image.shape[0], image.shape[1]))
            
            print(f"==== {image_type}图像检测完成 ====\n")
            
            return {
                'pose_keypoints': [
                    [float(kpt[0]), float(kpt[1]), float(score)]
                    for kpt, score in zip(keypoints, keypoint_scores)
                    if score > self.config.confidence_threshold
                ],
                'pose_subset': [[i for i in range(len(keypoints))]],
                'original_size': (image.shape[0], image.shape[1]),
                'scale': 1.0
            }
            
        except Exception as e:
            print(f"❌ {image_type}图像检测过程中发生错误: {str(e)}")
            logger.error(f"姿态检测失败: {str(e)}")
            logger.exception(e)  # 添加完整的堆栈跟踪以便更好地调试
            return self._empty_result((image.shape[0], image.shape[1]))
            
    def _empty_result(self, image_size: Tuple[int, int]) -> Dict:
        """返回空结果"""
        return {
            'pose_keypoints': [],
            'pose_subset': [],
            'original_size': image_size,
            'scale': 1.0
        }

class ImageProcessor:
    """图像处理器类"""
    def __init__(self, config: Config):
        self.config = config
        
    def extract_head(self, image: np.ndarray, landmarks: Dict) -> Tuple[np.ndarray, Dict]:
        """提取头部区域"""
        pose_points = landmarks['pose_keypoints']
        pose_subset = landmarks['pose_subset']
        
        height, width = image.shape[:2]
        
        # 如果没有检测到人或关键点不足
        if len(pose_subset) == 0 or len(pose_points) == 0:
            print("⚠️ 未检测到人体关键点，使用图像上部1/3作为头部区域")
            # 使用图像上部1/3作为头部区域
            head_height = height // 3
            center_x = width // 2
            
            position_info = {
                'top': 0,
                'left': max(0, center_x - head_height // 2),
                'bottom': head_height,
                'right': min(width, center_x + head_height // 2),
                'center_x': center_x,
                'center_y': head_height // 2
            }
            
            head_region = image[position_info['top']:position_info['bottom'], 
                          position_info['left']:position_info['right']]
            print(f"✅ 提取头部区域成功: 位置=[{position_info['left']},{position_info['top']},{position_info['right']},{position_info['bottom']}]")
            
            return head_region, position_info
        
        # 获取第一个检测到的人的关键点索引
        person_keypoints = pose_subset[0]
        
        # OpenPose关键点索引
        NECK = 1
        NOSE = 0
        
        # 尝试获取颈部和鼻子的关键点
        neck_idx = int(person_keypoints[NECK])
        nose_idx = int(person_keypoints[NOSE])
        
        # 如果没有检测到鼻子或颈部，使用其他可用关键点或图像上部
        if neck_idx == -1 and nose_idx == -1:
            # 使用图像上部1/3作为头部区域
            head_height = height // 3
            center_x = width // 2
            
            position_info = {
                'top': 0,
                'left': max(0, center_x - head_height // 2),
                'bottom': head_height,
                'right': min(width, center_x + head_height // 2),
                'center_x': center_x,
                'center_y': head_height // 2
            }
        elif neck_idx == -1:
            # 只有鼻子关键点
            nose_pos = pose_points[nose_idx]
            nose_x = int(nose_pos[0])
            nose_y = int(nose_pos[1])
            
            head_height = height // 4  # 估计头部高度
            head_width = int(head_height * 0.8)
            
            position_info = {
                'top': max(0, nose_y - head_height // 2),
                'left': max(0, nose_x - head_width // 2),
                'bottom': min(height, nose_y + head_height // 2),
                'right': min(width, nose_x + head_width // 2),
                'center_x': nose_x,
                'center_y': nose_y
            }
        elif nose_idx == -1:
            # 只有颈部关键点
            neck_pos = pose_points[neck_idx]
            neck_x = int(neck_pos[0])
            neck_y = int(neck_pos[1])
            
            head_height = height // 4  # 估计头部高度
            head_width = int(head_height * 0.8)
            
            position_info = {
                'top': max(0, neck_y - head_height),
                'left': max(0, neck_x - head_width // 2),
                'bottom': neck_y,
                'right': min(width, neck_x + head_width // 2),
                'center_x': neck_x,
                'center_y': neck_y - head_height // 2
            }
        else:
            # 有颈部和鼻子关键点
            neck_pos = pose_points[neck_idx]
            nose_pos = pose_points[nose_idx]
            
            neck_y = int(neck_pos[1])
            neck_x = int(neck_pos[0])
            nose_y = int(nose_pos[1])
            
            # 计算头部区域的边界框
            head_height = int((neck_y - nose_y) * 2.5)  # 留出足够空间
            head_width = int(head_height * 0.8)  # 假设头部宽高比约为0.8
            
            position_info = {
                'top': max(0, neck_y - head_height),
                'left': max(0, neck_x - head_width // 2),
                'bottom': min(height, neck_y),
                'right': min(width, neck_x + head_width // 2),
                'center_x': neck_x,
                'center_y': (neck_y + (neck_y - head_height)) // 2
            }
        
        head_region = image[position_info['top']:position_info['bottom'], 
                          position_info['left']:position_info['right']]
        
        print(f"✅ 提取头部区域成功: 位置=[{position_info['left']},{position_info['top']},{position_info['right']},{position_info['bottom']}]")
        
        return head_region, position_info

    def extract_body_parts(self, image: np.ndarray, landmarks: Dict) -> Dict[str, Dict]:
        """提取身体部位"""
        # 添加错误检查
        if not landmarks or 'pose_keypoints' not in landmarks or len(landmarks['pose_keypoints']) == 0:
            print("⚠️ 未检测到人体关键点，尝试基于图像进行估计")
            logger.warning("未检测到人体关键点，尝试基于图像进行估计")
            return self._estimate_body_parts(image)
        
        pose_points = landmarks['pose_keypoints']
        pose_subset = landmarks['pose_subset']
        
        if len(pose_subset) == 0:
            print("⚠️ 未检测到完整人物，尝试基于图像进行估计")
            logger.warning("未检测到完整人物，尝试基于图像进行估计")
            return self._estimate_body_parts(image)
        
        result = {}
        
        # OpenPose关键点索引
        BODY_PARTS = {
            'left_arm': [5, 6, 7],   # 左肩、左肘、左腕
            'right_arm': [2, 3, 4],  # 右肩、右肘、右腕
            'left_leg': [12, 13, 14],  # 左髋、左膝、左踝
            'right_leg': [9, 10, 11]   # 右髋、右膝、右踝
        }
        
        person_keypoints = pose_subset[0]
        
        for part_name, indices in BODY_PARTS.items():
            # 检查是否有该部位的至少一个关键点
            valid_indices = [i for i in indices if int(person_keypoints[i]) != -1]
            
            if valid_indices:
                # 有关键点，尝试提取部位
                try:
                    part_points = [pose_points[int(person_keypoints[i])] for i in valid_indices]
                    part_img, part_info = self._extract_part_by_points(image, part_points)
                    if part_img is not None and part_img.size > 0:
                        result[part_name] = {
                            'image': part_img,
                            'position': part_info
                        }
                        print(f"✅ 成功提取{part_name}: 位置=[{part_info['left']},{part_info['top']},{part_info['right']},{part_info['bottom']}]")
                except Exception as e:
                    print(f"❌ 提取{part_name}失败: {str(e)}")
                    logger.warning(f"提取{part_name}时发生错误: {str(e)}")
                    # 失败时使用估计方法
                    estimated_parts = self._estimate_specific_part(image, part_name)
                    if part_name in estimated_parts:
                        result[part_name] = estimated_parts[part_name]
                        print(f"⚠️ 使用估计方法提取{part_name}: 位置=[{result[part_name]['position']['left']},{result[part_name]['position']['top']},{result[part_name]['position']['right']},{result[part_name]['position']['bottom']}]")
            else:
                # 没有关键点，使用估计方法
                print(f"⚠️ 未检测到{part_name}的关键点，使用估计方法")
                estimated_parts = self._estimate_specific_part(image, part_name)
                if part_name in estimated_parts:
                    result[part_name] = estimated_parts[part_name]
                    print(f"⚠️ 使用估计方法提取{part_name}: 位置=[{result[part_name]['position']['left']},{result[part_name]['position']['top']},{result[part_name]['position']['right']},{result[part_name]['position']['bottom']}]")
        
        return result

    def _extract_part_by_points(self, image: np.ndarray, points: List) -> Tuple[np.ndarray, Dict]:
        """
        根据点列表提取身体部位
        
        参数:
        - image: 输入图像
        - points: 点列表，每个点是(x, y, confidence)形式
        
        返回:
        - 部位图像
        - 位置信息
        """
        height, width = image.shape[:2]
        
        # 提取x,y坐标
        xy_points = [(int(p[0]), int(p[1])) for p in points]
        
        # 创建边界框
        padding = 30
        min_x = max(0, min(p[0] for p in xy_points) - padding)
        max_x = min(width, max(p[0] for p in xy_points) + padding)
        min_y = max(0, min(p[1] for p in xy_points) - padding)
        max_y = min(height, max(p[1] for p in xy_points) + padding)
        
        # 计算中心点
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        position_info = {
            'top': min_y,
            'left': min_x,
            'bottom': max_y,
            'right': max_x,
            'center_x': center_x,
            'center_y': center_y,
            'points': xy_points
        }
        
        return image[min_y:max_y, min_x:max_x], position_info

    def _estimate_body_parts(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        在没有关键点的情况下，基于图像位置估计身体部位
        
        参数:
        - image: 输入图像
        
        返回:
        - 估计的身体部位字典
        """
        height, width = image.shape[:2]
        result = {}
        
        # 估计四个主要部位：左右手臂、左右腿
        parts_info = {
            'left_arm': {
                'x_start': 0, 
                'x_end': width // 2,
                'y_start': height // 4,
                'y_end': height // 2
            },
            'right_arm': {
                'x_start': width // 2, 
                'x_end': width,
                'y_start': height // 4,
                'y_end': height // 2
            },
            'left_leg': {
                'x_start': 0, 
                'x_end': width // 2,
                'y_start': height // 2,
                'y_end': height
            },
            'right_leg': {
                'x_start': width // 2, 
                'x_end': width,
                'y_start': height // 2,
                'y_end': height
            }
        }
        
        for part_name, area in parts_info.items():
            x_start = area['x_start']
            x_end = area['x_end']
            y_start = area['y_start']
            y_end = area['y_end']
            
            # 提取区域
            part_img = image[y_start:y_end, x_start:x_end]
            
            # 计算中心点
            center_x = (x_start + x_end) // 2
            center_y = (y_start + y_end) // 2
            
            position_info = {
                'top': y_start,
                'left': x_start,
                'bottom': y_end,
                'right': x_end,
                'center_x': center_x,
                'center_y': center_y,
                'points': [(center_x, center_y)]  # 使用中心点作为唯一的点
            }
            
            result[part_name] = {
                'image': part_img,
                'position': position_info
            }
        
        return result

    def _estimate_specific_part(self, image: np.ndarray, part_name: str) -> Dict[str, Dict]:
        """
        估计特定的身体部位
        
        参数:
        - image: 输入图像
        - part_name: 部位名称
        
        返回:
        - 估计的部位字典
        """
        height, width = image.shape[:2]
        result = {}
        
        # 基于部位名称确定区域
        if part_name == 'left_arm':
            x_start = 0
            x_end = width // 2
            y_start = height // 4
            y_end = height // 2
        elif part_name == 'right_arm':
            x_start = width // 2
            x_end = width
            y_start = height // 4
            y_end = height // 2
        elif part_name == 'left_leg':
            x_start = 0
            x_end = width // 2
            y_start = height // 2
            y_end = height
        elif part_name == 'right_leg':
            x_start = width // 2
            x_end = width
            y_start = height // 2
            y_end = height
        else:
            return result
        
        # 提取区域
        part_img = image[y_start:y_end, x_start:x_end]
        
        # 计算中心点
        center_x = (x_start + x_end) // 2
        center_y = (y_start + y_end) // 2
        
        position_info = {
            'top': y_start,
            'left': x_start,
            'bottom': y_end,
            'right': x_end,
            'center_x': center_x,
            'center_y': center_y,
            'points': [(center_x, center_y)]  # 使用中心点作为唯一的点
        }
        
        result[part_name] = {
            'image': part_img,
            'position': position_info
        }
        
        return result

    def calculate_placement_position(self, source_position: Dict, target_landmarks: Dict, part_name: str) -> Dict:
        """
        计算部位在目标图像中的放置位置
        
        参数:
        - source_position: 源图像中部位的位置信息
        - target_landmarks: 目标图像的关键点信息
        - part_name: 部位名称
        
        返回:
        - 包含x、y坐标的放置位置字典
        
        功能:
        - 根据部位类型(头部、手臂、腿部等)计算合适的放置位置
        - 确保部位与目标图像中对应部位位置对齐
        """
        target_pose = target_landmarks['pose_keypoints'][0]
        
        # 根据不同部位计算放置位置
        placement = {}
        
        if part_name == "Head":
            # 对于头部，使用目标图像的颈部和鼻子位置
            NECK = 1
            NOSE = 0
            neck_pos = target_pose[NECK]
            nose_pos = target_pose[NOSE]
            
            target_neck_y = int(neck_pos[1])
            target_neck_x = int(neck_pos[0])
            
            # 计算头部区域
            head_height = int((neck_pos[1] - nose_pos[1]) * 2.5)
            
            placement = {
                'x': target_neck_x - (source_position['right'] - source_position['left']) // 2,
                'y': target_neck_y - head_height
            }
        elif "arm" in part_name:
            # 对于手臂，使用相应的肩、肘、腕关键点
            if part_name == "left_arm":
                indices = [5, 6, 7]  # 左肩、左肘、左腕
            else:
                indices = [2, 3, 4]  # 右肩、右肘、右腕
                
            target_points = [(int(target_pose[i][0]), int(target_pose[i][1])) 
                            for i in indices if target_pose[i][2] > 0.1]
            
            if target_points:
                # 计算目标区域中心
                target_center_x = sum(p[0] for p in target_points) // len(target_points)
                target_center_y = sum(p[1] for p in target_points) // len(target_points)
                
                # 计算放置位置，使部位中心与目标中心对齐
                placement = {
                    'x': target_center_x - (source_position['right'] - source_position['left']) // 2,
                    'y': target_center_y - (source_position['bottom'] - source_position['top']) // 2
                }
        elif "leg" in part_name:
            # 对于腿部，使用相应的髋、膝、踝关键点
            if part_name == "left_leg":
                indices = [12, 13, 14]  # 左髋、左膝、左踝
            else:
                indices = [9, 10, 11]   # 右髋、右膝、右踝
                
            target_points = [(int(target_pose[i][0]), int(target_pose[i][1])) 
                            for i in indices if target_pose[i][2] > 0.1]
            
            if target_points:
                # 计算目标区域中心
                target_center_x = sum(p[0] for p in target_points) // len(target_points)
                target_center_y = sum(p[1] for p in target_points) // len(target_points)
                
                # 计算放置位置，使部位中心与目标中心对齐
                placement = {
                    'x': target_center_x - (source_position['right'] - source_position['left']) // 2,
                    'y': target_center_y - (source_position['bottom'] - source_position['top']) // 2
                }
        
        return placement

class CharacterAutoMerger:
    """角色自动合成器"""
    def __init__(self, dwpose_dir: str = r"D:\drafts\DWPose"):
        self.config = Config(dwpose_dir)
        self.pose_detector = PoseDetector(self.config)
        self.image_processor = ImageProcessor(self.config)
        logger.info("CharacterAutoMerger 初始化成功")

    def process_images(self, template_path: str, head_path: str, body_path: str, output_path: str = "output.psd") -> str:
        """处理图像"""
        try:
            print("\n========== 开始处理图像 ==========")
            print(f"模板图像: {template_path}")
            print(f"头部图像: {head_path}")
            print(f"身体图像: {body_path}")
            print(f"输出路径: {output_path}")
            
            # 加载图像
            template = cv2.imread(template_path)
            head_image = cv2.imread(head_path)
            body_image = cv2.imread(body_path)
            
            print(f"\n模板图像尺寸: {template.shape[1]}x{template.shape[0]}")
            print(f"头部图像尺寸: {head_image.shape[1]}x{head_image.shape[0]}")
            print(f"身体图像尺寸: {body_image.shape[1]}x{body_image.shape[0]}")
            
            # 获取姿态信息
            print("\n检测各图像姿态信息...")
            template_landmarks = self.pose_detector.detect_pose(template, "模板")
            head_landmarks = self.pose_detector.detect_pose(head_image, "头部")
            body_landmarks = self.pose_detector.detect_pose(body_image, "身体")
            
            # 提取部位
            print("\n开始提取头部和身体部位...")
            head, head_position = self.image_processor.extract_head(head_image, head_landmarks)
            body_parts = self.image_processor.extract_body_parts(body_image, body_landmarks)
            
            print(f"\n共提取了 {1 + len(body_parts)} 个部位:")
            print(f"- 头部: {head.shape[1]}x{head.shape[0]}")
            for part_name, part_info in body_parts.items():
                part_img = part_info['image']
                print(f"- {part_name}: {part_img.shape[1]}x{part_img.shape[0]}")
            
            # 合成图像
            print("\n开始进行图像合成...")
            if self._has_photoshop():
                result = self._merge_with_photoshop(template, head, head_position, body_parts, template_landmarks, output_path)
                print(f"\n✅ 图像合成完成: {result}")
                print("========== 处理完成 ==========\n")
                return result
            else:
                print("❌ 未检测到PSD-Tools，无法进行合成")
                raise RuntimeError("未检测到PSD-Tools，无法进行合成")
                
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            logging.error(f"处理失败: {str(e)}")
            raise
            
    def _has_photoshop(self) -> bool:
        """检查是否可以使用PSD-Tools"""
        try:
            import psd_tools
            return True
        except ImportError:
            return False
            
    def _merge_with_photoshop(self, template: np.ndarray, head: np.ndarray, head_position: Dict, 
                             body_parts: Dict, template_landmarks: Dict, output_path: str) -> str:
        """使用PSD-Tools合成图像"""
        try:
            from psd_tools import PSDImage
            from psd_tools.constants import ColorMode
            
            # 创建临时目录存放图像文件
            temp_dir = tempfile.gettempdir()
            
            # 创建新的PSD文档 - 修正参数
            psd = PSDImage.new(
                'RGB',  # 使用字符串 'RGB' 而不是 ColorMode.RGB
                (template.shape[1], template.shape[0])  # 第二个位置参数是 size，以元组形式传递
            )
            
            # 添加模板图层
            template_path = os.path.join(temp_dir, 'template.png')
            # 转换BGR到RGB并保存临时文件
            cv2.imwrite(template_path, cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
            template_img = Image.open(template_path)
            # 使用 create_layer 替代 add_layer
            template_layer = psd.create_layer(name="Template")
            template_layer.composite(template_img, (0, 0))
            
            # 添加头部图层
            head_path = os.path.join(temp_dir, 'head.png')
            cv2.imwrite(head_path, cv2.cvtColor(head, cv2.COLOR_BGR2RGB))
            head_img = Image.open(head_path)
            
            # 计算头部放置位置
            head_placement = self.image_processor.calculate_placement_position(
                head_position, template_landmarks, "Head"
            )
            
            # 添加头部图层并设置位置
            # 使用 create_layer 替代 add_layer
            head_layer = psd.create_layer(name="Head")
            head_x = int(head_placement.get('x', 0))
            head_y = int(head_placement.get('y', 0))
            # 将图像合成到图层上，并设置位置
            head_layer.composite(head_img, (head_x, head_y))
            
            # 添加身体部位图层
            for part_name, part_info in body_parts.items():
                part_img = part_info['image']
                part_position = part_info['position']
                
                # 保存临时图像文件
                part_path = os.path.join(temp_dir, f'{part_name}.png')
                cv2.imwrite(part_path, cv2.cvtColor(part_img, cv2.COLOR_BGR2RGB))
                part_pil_img = Image.open(part_path)
                
                # 计算部位放置位置
                placement = self.image_processor.calculate_placement_position(
                    part_position, template_landmarks, part_name
                )
                
                # 添加图层并设置位置
                # 使用 create_layer 替代 add_layer
                part_layer = psd.create_layer(name=part_name)
                part_x = int(placement.get('x', 0))
                part_y = int(placement.get('y', 0))
                # 将图像合成到图层上，并设置位置
                part_layer.composite(part_pil_img, (part_x, part_y))
            
            # 保存PSD文件
            psd.save(output_path)
            
            # 清理临时文件
            for temp_file in [template_path, head_path] + [os.path.join(temp_dir, f'{part}.png') for part in body_parts]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            return output_path
            
        except Exception as e:
            logger.error(f"PSD-Tools合成失败: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        # 配置参数
        SETUP_ENV = False  # 是否需要设置环境
        DWPOSE_DIR = r"D:\drafts\DWPose"
        
        # 配置输入和输出路径
        TEMPLATE_PATH = r"D:\drafts\CAM\template\teen girl template.png"
        HEAD_PATH = r"D:\drafts\CAM\template\head_raw.png"
        BODY_PATH = r"D:\drafts\CAM\template\body_raw.png"
        OUTPUT_PATH = r"D:\drafts\CAM\output\merged_character.psd"  # 指定输出PSD文件路径
        
        # 确保输出目录存在
        output_dir = os.path.dirname(OUTPUT_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"创建输出目录: {output_dir}")
        
        # 如果需要设置环境
        if SETUP_ENV:
            env_manager = EnvironmentManager(DWPOSE_DIR)
            try:
                env_manager.setup()
                logger.info("环境设置完成")
            except Exception as e:
                logger.error(f"环境设置失败: {str(e)}")
                return 1

        # 初始化合成器并处理图像
        merger = CharacterAutoMerger(DWPOSE_DIR)
        result = merger.process_images(
            template_path=TEMPLATE_PATH,
            head_path=HEAD_PATH,
            body_path=BODY_PATH,
            output_path=OUTPUT_PATH  # 使用配置的输出路径
        )
        print(f"处理完成: {result}")
        return 0
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
