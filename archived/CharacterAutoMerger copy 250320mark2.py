import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
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
import photoshop.api as ps
import copy

'''
# 程序概述
本程序(CharacterAutoMerger)是一个角色自动合成工具，使用姿态估计技术将不同图像中的角色部位(如头部、手臂、腿部等)
智能地合成到目标模板中。主要依赖OpenPose进行人体关键点检测，通过对应关键点之间的映射实现智能部位合成。

# 代码结构
1. 初始化与环境设置
   - 日志配置
   - 依赖包检查与安装
   - OpenPose模型加载

2. CharacterAutoMerger类
   - 初始化模型和参数
   - 姿态检测功能
   - 身体部位提取功能
   - 图像合成与Photoshop交互功能

3. 主函数
   - 参数设置与验证
   - 执行图像处理流程

# 处理流程
1. 加载模型和输入图像(模板图像、头部图像、身体图像)
2. 对所有图像进行姿态关键点检测
3. 从头部图像提取头部区域
4. 从身体图像提取各个身体部位(手臂、腿部等)
5. 计算各部位在模板图像中的放置位置
6. 使用Photoshop API将各部位合成到模板图像中
7. 保存为PSD文件

# 依赖
- pytorch-openpose: 人体姿态检测
- photoshop-python-api: 与Photoshop的交互
- OpenCV, PIL: 图像处理
- PyTorch: 深度学习框架
'''

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_openpose():
    """
    设置OpenPose环境
    
    功能：
    - 检查并添加pytorch-openpose目录到Python路径
    - 检查src目录结构是否完整
    - 创建必要的__init__.py文件
    - 导入所需的OpenPose模块
    
    返回：
    - 成功导入的模块元组(Body, Hand, util) 或 None（如果失败）
    """
    # 获取pytorch-openpose的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pytorch_openpose_dir = os.path.join('D:\\', 'drafts', 'pytorch-openpose')

    # 添加pytorch-openpose到Python路径
    if pytorch_openpose_dir not in sys.path:
        sys.path.insert(0, pytorch_openpose_dir)  # 将路径添加到最前面，确保优先使用本地包

    # 检查src目录是否存在
    src_dir = os.path.join(pytorch_openpose_dir, 'src')
    if not os.path.exists(src_dir):
        logger.error(f"src目录不存在: {src_dir}")
        return None

    # 检查__init__.py文件
    init_file = os.path.join(src_dir, '__init__.py')
    if not os.path.exists(init_file):
        logger.warning(f"创建src包的__init__.py文件: {init_file}")
        try:
            with open(init_file, 'w') as f:
                pass  # 创建空的__init__.py文件
        except Exception as e:
            logger.error(f"创建__init__.py文件失败: {str(e)}")
            return None

    try:
        from src.body import Body
        from src.hand import Hand
        from src import util
        logger.info("成功导入pytorch-openpose模块")
        return Body, Hand, util
    except ImportError as e:
        logger.error(f"无法导入pytorch-openpose模块: {str(e)}")
        logger.error(f"请确保pytorch-openpose已正确安装在: {pytorch_openpose_dir}")
        logger.error(f"并且src目录结构完整: {src_dir}")
        return None

# 设置OpenPose并导入所需模块
openpose_modules = setup_openpose()
if not openpose_modules:
    sys.exit(1)
else:
    Body, Hand, util = openpose_modules

def install_dependencies():
    """
    检查并安装所需的Python依赖包
    
    功能:
    - 检查numpy, opencv-python, pillow, photoshop-python-api, torch, torchvision是否已安装
    - 如果发现缺失的包，自动使用pip进行安装
    
    异常:
    - 如果安装失败，记录错误并抛出异常
    """
    logger.info("检查依赖包...")
    required = {
        'numpy', 
        'opencv-python', 
        'pillow', 
        'photoshop-python-api',
        'torch',
        'torchvision'
    }
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if (missing):
        logger.info(f"发现缺失的依赖包: {missing}")
        try:
            python = sys.executable
            subprocess.check_call([python, '-m', 'pip', 'install', *missing], 
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
            logger.info("依赖包安装完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"安装依赖包失败: {str(e)}")
            raise

# 在文件开头添加依赖检查
try:
    install_dependencies()
except Exception as e:
    logger.error(f"安装依赖包时出错: {str(e)}")
    sys.exit(1)

class CharacterAutoMerger:
    """
    角色自动合成器类
    
    主要功能:
    - 使用OpenPose进行姿态检测
    - 提取和识别人物的各个部位
    - 将部位智能地合成到模板图像中
    """
    
    def __init__(self, models_path=None):
        """
        初始化合成器
        
        参数:
        - models_path: OpenPose模型文件的路径，默认为D:/drafts/pytorch-openpose/model
        
        功能:
        - 初始化身体和手部姿态估计模型
        - 设置推理参数(stride, upsample_ratio等)
        
        注意:
        - 使用安全模式加载模型(weights_only=True)以提高兼容性
        """
        if models_path is None:
            models_path = os.path.join('D:\\', 'drafts', 'pytorch-openpose', 'model')
        # 初始化PyTorch版本的OpenPose
        try:
            # 修改Body和Hand初始化时传递weights_only=True参数
            self.body_model_path = os.path.join(models_path, 'body_pose_model.pth')
            self.hand_model_path = os.path.join(models_path, 'hand_pose_model.pth')
            
            # 自定义加载模型的方式，确保使用weights_only=True
            from src.body import Body as OriginalBody
            from src.hand import Hand as OriginalHand
            
            class SafeBody(OriginalBody):
                def __init__(self, model_path):
                    super().__init__(model_path)
                    # 重新加载模型，使用weights_only=True
                    model_dict = util.transfer(self.model, torch.load(model_path, weights_only=True))
                    self.model.load_state_dict(model_dict)
            
            class SafeHand(OriginalHand):
                def __init__(self, model_path):
                    super().__init__(model_path)
                    # 重新加载模型，使用weights_only=True
                    model_dict = util.transfer(self.model, torch.load(model_path, weights_only=True))
                    self.model.load_state_dict(model_dict)
            
            self.body_estimation = SafeBody(self.body_model_path)
            self.hand_estimation = SafeHand(self.hand_model_path)
            
            logger.info("PyTorch OpenPose初始化成功（已启用安全加载模式）")
        except Exception as e:
            logger.error(f"初始化PyTorch OpenPose失败: {str(e)}")
            raise
            
        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = 18  # OpenPose默认关键点数量
        self.height_size = 256  # 添加默认的height_size参数

    def normalize(self, img, mean, scale):
        """
        图像归一化处理
        
        参数:
        - img: 输入图像
        - mean: 均值
        - scale: 缩放因子
        
        返回:
        - 归一化后的图像
        """
        img = np.array(img, dtype=np.float32)
        img = (img - mean) * scale
        return img

    def pad_width(self, img, stride, pad_value, min_dims):
        """
        填充图像到特定尺寸
        
        参数:
        - img: 输入图像
        - stride: 步长
        - pad_value: 填充值
        - min_dims: 最小尺寸
        
        返回:
        - 填充后的图像和填充信息
        """
        h, w = img.shape[:2]
        h = min(min_dims[0], h)
        min_dims[0] = ((min_dims[0] + stride - 1) // stride) * stride
        min_dims[1] = ((min_dims[1] + stride - 1) // stride) * stride
        pad = []
        pad.append(int(min(stride - (h % stride), stride - 1)))
        pad.append(int(min(min_dims[1] - w, stride - 1)))
        
        img_padded = cv2.copyMakeBorder(img, 0, pad[0], 0, pad[1],
                                      cv2.BORDER_CONSTANT, value=pad_value)
        return img_padded, pad

    def infer_fast(self, img, height_size=256):
        """
        快速姿态推理
        
        参数:
        - img: 输入图像
        - height_size: 缩放后的高度
        
        返回:
        - heatmaps: 热图
        - pafs: 部位亲和场
        - scale: 缩放比例
        - pad: 填充信息
        
        功能:
        - 图像预处理(缩放、归一化、填充)
        - 运行神经网络推理
        - 处理网络输出为热图和部位亲和场
        """
        height, width, _ = img.shape
        scale = height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = self.normalize(scaled_img, np.array([128, 128, 128], np.float32), np.float32(1/256))
        min_dims = [height_size, max(scaled_img.shape[1], height_size)]
        padded_img, pad = self.pad_width(scaled_img, self.stride, (0, 0, 0), min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if torch.cuda.is_available():
            tensor_img = tensor_img.cuda()

        # 使用body_estimation进行推理
        with torch.no_grad():
            stages_output = self.body_estimation(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    def get_pose_landmarks(self, image: np.ndarray) -> Dict:
        """
        获取图像中的姿态关键点
        
        参数:
        - image: 输入图像
        
        返回:
        - 包含姿态关键点和手部关键点的字典
        
        功能:
        - 使用OpenPose检测人体关键点
        - 检测手部关键点
        - 合并结果并返回
        """
        try:
            # 运行姿态检测
            candidate, subset = self.body_estimation(image)
            
            # 添加调试信息
            logger.debug(f"检测到的关键点数量: {len(candidate)}")
            logger.debug(f"检测到的人物数量: {len(subset)}")
            
            if len(candidate) == 0:
                logger.warning("未检测到任何关键点")
            if len(subset) == 0:
                logger.warning("未检测到任何人物")
            
            # 检测手部
            hands_list = util.handDetect(candidate, subset, image)
            all_hand_peaks = []
            
            for x, y, w, is_left in hands_list:
                peaks = self.hand_estimation(image[y:y+w, x:x+w, :])
                peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                all_hand_peaks.append(peaks)

            return {
                'pose_keypoints': candidate,
                'pose_subset': subset,
                'hand_peaks': all_hand_peaks
            }
        except Exception as e:
            logger.error(f"姿态检测失败: {str(e)}")
            raise

    def extract_head(self, image: np.ndarray, landmarks: Dict) -> Tuple[np.ndarray, Dict]:
        """
        提取图像中的头部区域
        
        参数:
        - image: 输入图像
        - landmarks: 姿态关键点信息
        
        返回:
        - 头部区域图像
        - 头部位置信息字典
        
        功能:
        - 根据颈部和鼻子关键点确定头部区域
        - 如果关键点检测不完整，会尝试使用图像中心作为参考
        - 提取相应区域并返回位置信息
        """
        pose_points = landmarks['pose_keypoints']
        pose_subset = landmarks['pose_subset']
        
        # 修改：降低判断标准，即使没有完整的姿态也尝试处理
        if len(pose_subset) == 0:
            # 尝试使用图像中心作为参考点
            height, width = image.shape[:2]
            center_x = width // 2
            center_y = height // 2
            
            # 创建一个基本的头部区域
            head_height = height // 3  # 假设头部占图像高度的1/3
            head_width = int(head_height * 0.8)
            
            position_info = {
                'top': center_y - head_height,
                'left': center_x - head_width // 2,
                'bottom': center_y + head_height,
                'right': center_x + head_width // 2,
                'center_x': center_x,
                'center_y': center_y
            }
            
            return image[position_info['top']:position_info['bottom'], 
                        position_info['left']:position_info['right']], position_info
        
        # 获取第一个检测到的人的关键点索引
        person_keypoints = pose_subset[0]
        
        # OpenPose关键点索引
        NECK = 1
        NOSE = 0
        
        # 获取颈部和鼻子的关键点
        neck_idx = int(person_keypoints[NECK])
        nose_idx = int(person_keypoints[NOSE])
        
        if neck_idx == -1 or nose_idx == -1:
            raise ValueError("未检测到颈部或鼻子关键点")
            
        neck_pos = pose_points[neck_idx]
        nose_pos = pose_points[nose_idx]
        
        height, width = image.shape[:2]
        neck_y = int(neck_pos[1])
        neck_x = int(neck_pos[0])
        nose_y = int(nose_pos[1])
        
        # 计算头部区域的边界框
        head_height = int((neck_y - nose_y) * 2.5)  # 留出足够空间
        head_width = int(head_height * 0.8)  # 假设头部宽高比约为0.8
        
        # 提取头部区域
        top = max(0, neck_y - head_height)
        left = max(0, neck_x - head_width // 2)
        bottom = min(height, neck_y)
        right = min(width, neck_x + head_width // 2)
        
        head_region = image[top:bottom, left:right]
        
        # 返回头部区域和位置信息
        position_info = {
            'top': top,
            'left': left,
            'bottom': bottom,
            'right': right,
            'center_x': neck_x,
            'center_y': (top + bottom) // 2
        }
        
        return head_region, position_info

    def extract_body_parts(self, image: np.ndarray, landmarks: Dict) -> Dict[str, Dict]:
        """
        提取图像中的身体各部分
        
        参数:
        - image: 输入图像
        - landmarks: 姿态关键点信息
        
        返回:
        - 包含各身体部位图像和位置信息的字典
        
        功能:
        - 提取左右手臂、左右腿部等身体部位
        - 为每个部位计算位置信息
        """
        # 添加错误检查
        if not landmarks or 'pose_keypoints' not in landmarks or len(landmarks['pose_keypoints']) == 0:
            logger.warning("未检测到人体关键点，返回空结果")
            return {}
        
        try:
            pose_points = landmarks['pose_keypoints']
            # 如果pose_points是二维数组，取第一个人的数据
            if len(pose_points.shape) > 1:
                pose_points = pose_points[0]
        except IndexError:
            logger.warning("姿态关键点数据格式不正确，返回空结果")
            return {}
        
        result = {}
        
        # OpenPose关键点索引
        BODY_PARTS = {
            'left_arm': [5, 6, 7],   # 左肩、左肘、左腕
            'right_arm': [2, 3, 4],  # 右肩、右肘、右腕
            'left_leg': [12, 13, 14],  # 左髋、左膝、左踝
            'right_leg': [9, 10, 11]   # 右髋、右膝、右踝
        }
        
        for part_name, indices in BODY_PARTS.items():
            try:
                part_img, part_info = self._extract_limb(image, pose_points, indices)
                if part_img is not None and part_img.size > 0:
                    result[part_name] = {
                        'image': part_img,
                        'position': part_info
                    }
            except Exception as e:
                logger.warning(f"提取{part_name}时发生错误: {str(e)}")
                continue
            
        return result

    def _extract_limb(self, image: np.ndarray, pose_points: np.ndarray, keypoint_indices: List) -> Tuple[np.ndarray, Dict]:
        """
        提取单个肢体部位
        
        参数:
        - image: 输入图像
        - pose_points: 姿态关键点
        - keypoint_indices: 相关关键点索引
        
        返回:
        - 肢体部位图像
        - 位置信息字典
        
        功能:
        - 根据指定关键点确定肢体区域
        - 提取相应区域并返回位置信息
        """
        height, width = image.shape[:2]
        points = []
        
        for idx in keypoint_indices:
            point = pose_points[idx]
            if point[2] > 0.1:  # 检查置信度
                x = int(point[0])
                y = int(point[1])
                points.append((x, y))
                
        if not points:
            return np.array([]), {}
            
        # 创建肢体周围的边界框
        padding = 30
        min_x = max(0, min(p[0] for p in points) - padding)
        max_x = min(width, max(p[0] for p in points) + padding)
        min_y = max(0, min(p[1] for p in points) - padding)
        max_y = min(height, max(p[1] for p in points) + padding)
        
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
            'points': points
        }
        
        return image[min_y:max_y, min_x:max_x], position_info

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

    def merge_to_template(self, template: np.ndarray, head: np.ndarray, head_position: Dict,
                         body_parts: Dict[str, Dict], template_landmarks: Dict) -> str:
        """
        将所有部分合并到模板中并保存为PSD
        
        参数:
        - template: 模板图像
        - head: 头部图像
        - head_position: 头部位置信息
        - body_parts: 身体部位信息字典
        - template_landmarks: 模板图像的关键点信息
        
        返回:
        - 输出PSD文件的路径
        
        功能:
        - 使用Photoshop API创建新文档
        - 添加模板图层
        - 添加头部和各身体部位的图层
        - 根据计算的位置放置各部位
        - 保存为PSD文件
        """
        try:
            app = ps.Application()
        except Exception as e:
            logger.error("无法初始化Photoshop. 请确保已安装Photoshop并配置了photoshop-python-api")
            raise e

        # 创建新文档
        doc = app.documents.add(template.shape[1], template.shape[0])
        
        # 添加模板图层
        template_layer = doc.artLayers.add()
        template_layer.name = "Template"
        template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        template_pil = Image.fromarray(template_rgb)
        doc.paste()  # 移除参数
        
        # 添加头部图层
        if head is not None and head.size > 0:
            head_layer = doc.artLayers.add()
            head_layer.name = "Head"
            head_rgb = cv2.cvtColor(head, cv2.COLOR_BGR2RGB)
            head_pil = Image.fromarray(head_rgb)
            
            # 计算头部放置位置
            head_placement = self.calculate_placement_position(
                head_position, template_landmarks, "Head"
            )
            
            # 在Photoshop中放置头部
            doc.activeLayer = head_layer
            doc.paste()  # 移除参数
            # 移动图层到计算出的位置
            try:
                head_layer.translate(head_placement['x'], head_placement['y'])
            except:
                logger.error(f"无法移动头部图层到位置 ({head_placement['x']}, {head_placement['y']})")
        
        # 添加身体部分
        for part_name, part_data in body_parts.items():
            part_image = part_data['image']
            part_position = part_data['position']
            
            if part_image is not None and part_image.size > 0:
                layer = doc.artLayers.add()
                layer.name = part_name
                part_rgb = cv2.cvtColor(part_image, cv2.COLOR_BGR2RGB)
                part_pil = Image.fromarray(part_rgb)
                
                # 计算部位放置位置
                placement = self.calculate_placement_position(
                    part_position, template_landmarks, part_name
                )
                
                # 在Photoshop中放置部位
                doc.activeLayer = layer
                doc.paste()  # 移除参数
                # 移动图层到计算出的位置
                try:
                    layer.translate(placement['x'], placement['y'])
                except:
                    logger.error(f"无法移动{part_name}图层到位置 ({placement['x']}, {placement['y']})")
        
        # 保存PSD文件
        output_path = "output.psd"
        doc.saveAs(output_path, True)  # 添加True作为第二个参数
        doc.close()
        
        return output_path

    def process_images(self, template_path: str, head_image_path: str, body_image_path: str, output_path: str = "output.psd") -> str:
        """
        处理主流程，将头部和身体图像合成到模板中
        
        参数:
        - template_path: 模板图像的路径
        - head_image_path: 头部图像的路径
        - body_image_path: 身体图像的路径
        - output_path: 输出PSD文件的路径
        
        返回:
        - 输出文件的路径
        
        功能:
        - 读取输入图像
        - 获取所有图像的姿态信息
        - 提取头部和身体部位
        - 合并图像并保存为PSD
        - 清理内存并返回结果
        
        异常:
        - ValueError: 当图像读取失败或关键点检测失败时
        - RuntimeError: 当处理过程中出现严重错误时
        """
        try:
            # 读取图像并进行基本验证
            images = {
                'template': ('模板', template_path),
                'head': ('头部', head_image_path),
                'body': ('身体', body_image_path)
            }
            
            loaded_images = {}
            for key, (name, path) in images.items():
                logger.info(f"正在读取{name}图像: {path}")
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"无法读取{name}图像: {path}")
                if img.size == 0:
                    raise ValueError(f"{name}图像是空的: {path}")
                loaded_images[key] = img

            # 使用生成器处理姿态检测，提高内存效率
            def process_landmarks():
                for key, (name, img) in zip(
                    ['template', 'head', 'body'],
                    [('模板', loaded_images['template']),
                     ('头部', loaded_images['head']),
                     ('身体', loaded_images['body'])]
                ):
                    logger.info(f"正在分析{name}图像姿态...")
                    landmarks = self.get_pose_landmarks(img)
                    if landmarks is None:
                        raise ValueError(f"无法在{name}图像中检测到姿态关键点")
                    yield key, landmarks

            # 获取所有图像的姿态信息
            landmarks_dict = {k: v for k, v in process_landmarks()}

            # 提取头部
            logger.info("正在提取头部区域...")
            head, head_position = self.extract_head(loaded_images['head'], landmarks_dict['head'])
            if head is None or head.size == 0:
                raise ValueError("头部提取失败")

            # 提取身体部分
            logger.info("正在提取身体部位...")
            body_parts = self.extract_body_parts(loaded_images['body'], landmarks_dict['body'])
            if not body_parts:
                raise ValueError("未能提取出任何身体部位")

            # 合并图像
            logger.info("正在合成最终图像...")
            try:
                result_path = self.merge_to_template(
                    loaded_images['template'],
                    head,
                    head_position,
                    body_parts,
                    landmarks_dict['template']
                )
            except Exception as e:
                raise RuntimeError(f"图像合成失败: {str(e)}")

            # 清理内存
            for img in loaded_images.values():
                del img
            del landmarks_dict, head, body_parts

            logger.info(f"处理完成，已保存到: {result_path}")
            return result_path

        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            raise

def main():
    """
    主函数，程序入口点
    
    功能:
    - 检查环境和模型文件
    - 初始化CharacterAutoMerger实例
    - 设置输入参数(模板、头部、身体图像路径)
    - 执行图像处理流程
    - 捕获和记录可能的异常
    
    返回:
    - 0: 处理成功
    - 1: 处理失败
    """
    try:
        logger.info("程序启动")
        
        # 获取pytorch-openpose的路径
        pytorch_openpose_dir = os.path.join('D:\\', 'drafts', 'pytorch-openpose')
        
        # 检查pytorch-openpose目录是否存在
        if not os.path.exists(pytorch_openpose_dir):
            logger.error(f"pytorch-openpose目录不存在: {pytorch_openpose_dir}")
            return 1
            
        # 检查模型文件
        model_dir = os.path.join(pytorch_openpose_dir, "model")
        body_model_path = os.path.join(model_dir, "body_pose_model.pth")
        hand_model_path = os.path.join(model_dir, "hand_pose_model.pth")
        
        for model_path in [body_model_path, hand_model_path]:
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return 1
            
        # 创建合成器实例
        logger.info("初始化CharacterAutoMerger...")
        merger = CharacterAutoMerger()  # 不需要显式指定models_path，使用默认值
        
        # 在脚本中直接定义参数，而非通过命令行传入
        # 修改这些路径为你的实际图像路径
        # template_path = r"D:\drafts\CAM\template\teen girl template.png"#注意 不能用中文文件名
        # head_path = r"D:\drafts\CAM\template\image (12).png"
        # body_path = r"D:\drafts\CAM\template\output_image.png"
        # output_path = r"D:\drafts\CAM\template\output.psd"

        template_path = r"D:\drafts\CAM\template\teen girl template.png"#注意 不能用中文文件名
        head_path = r"D:\drafts\pytorch-openpose\images\demo.jpg"
        body_path = r"D:\drafts\pytorch-openpose\images\demo.jpg"
        output_path = r"D:\drafts\CAM\template\output.psd"
        
        logger.info(f"使用的参数: template={template_path}, head={head_path}, body={body_path}, output={output_path}")
        
        # 检查输入文件是否存在
        for path, name in [
            (template_path, "模板"),
            (head_path, "头部"),
            (body_path, "身体")
        ]:
            if not os.path.exists(path):
                logger.error(f"{name}图像路径不存在: {path}")
                return 1
            else:
                logger.info(f"找到{name}图像: {path}")
        
        # 处理图像
        logger.info("开始处理图像...")
        output_file = merger.process_images(
            template_path,
            head_path,
            body_path,
            output_path
        )
        
        logger.info(f"处理完成! 输出文件: {output_file}")
        return 0
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    logger.info(f"程序结束，退出代码: {exit_code}")
    sys.exit(exit_code)
