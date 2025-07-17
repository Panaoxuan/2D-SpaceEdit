import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import os
import requests
from tqdm import tqdm

def download_sam_checkpoint(save_path, url):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading SAM checkpoint from {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=save_path) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Downloaded to {save_path}")

def ensure_sam_checkpoint():
    ckpt_path = "/home/liushuzhi/pax/2.5d_editing/checkpoints/sam/sam_vit_h_4b8939.pth"
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    if not os.path.exists(ckpt_path):
        print(f"SAM checkpoint not found at {ckpt_path}. Downloading...")
        download_sam_checkpoint(ckpt_path, url)
    else:
        print(f"SAM checkpoint found at {ckpt_path}.")

class SimplifiedSAMProcessor:
    def __init__(self, model_type="vit_h", checkpoint_path="/home/liushuzhi/pax/2.5d_editing/checkpoints/sam/sam_vit_h_4b8939.pth", max_display_size=800, output_base_dir="/home/liushuzhi/pax/2.5d_editing/outputs"):
        """初始化SAM模型"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        
        # 创建预测器
        self.predictor = SamPredictor(self.sam)
        
        # 存储当前状态
        self.original_image = None
        self.current_mask = None
        self.points = []
        self.labels = []
        
        # 显示相关
        self.max_display_size = max_display_size
        self.scale_factor = 1.0
        self.display_image = None
        
        # 新增：mask扩张参数
        self.dilation_kernel_size = 15  # 默认扩张核大小
        self.dilation_iterations = 1    # 扩张迭代次数
        
        # 输出目录配置
        self.output_base_dir = output_base_dir
        self.output_dirs = {
            'objects': os.path.join(output_base_dir, 'objects'),
            'holes': os.path.join(output_base_dir, 'holes'),
            'mask_dilated': os.path.join(output_base_dir, 'mask_dilated'),
            'mask_precise': os.path.join(output_base_dir, 'mask_precise'),
            'mask_edge': os.path.join(output_base_dir, 'mask_edge')  # 新增：扩张边缘mask目录
        }
        
        # 创建输出目录
        self.create_output_directories()
        
    def create_output_directories(self):
        """创建输出目录"""
        for dir_name, dir_path in self.output_dirs.items():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"✓ 创建目录: {dir_path}")
            else:
                print(f"✓ 目录已存在: {dir_path}")
    
    def resize_for_display(self, image):
        """调整图像到合适的显示尺寸"""
        h, w = image.shape[:2]
        
        if max(h, w) <= self.max_display_size:
            self.scale_factor = 1.0
            return image
        
        self.scale_factor = self.max_display_size / max(h, w)
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"图像缩放: {w}x{h} -> {new_w}x{new_h} (缩放比例: {self.scale_factor:.2f})")
        
        return resized
    
    def display_coord_to_original(self, x, y):
        """将显示坐标转换为原图坐标"""
        return int(x / self.scale_factor), int(y / self.scale_factor)
    
    def original_coord_to_display(self, x, y):
        """将原图坐标转换为显示坐标"""
        return int(x * self.scale_factor), int(y * self.scale_factor)
    
    def dilate_mask(self, mask, kernel_size=None, iterations=None):
        """
        扩张mask以改善inpainting效果
        
        Args:
            mask: 原始mask (bool array)
            kernel_size: 扩张核大小
            iterations: 迭代次数
            
        Returns:
            dilated_mask: 扩张后的mask
        """
        if kernel_size is None:
            kernel_size = self.dilation_kernel_size
        if iterations is None:
            iterations = self.dilation_iterations
            
        # 转换为uint8格式
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # 创建扩张核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 执行形态学扩张
        dilated_uint8 = cv2.dilate(mask_uint8, kernel, iterations=iterations)
        
        # 转换回bool
        dilated_mask = dilated_uint8 > 128
        
        return dilated_mask
    
    def get_edge_mask(self, original_mask, kernel_size=None, iterations=None):
        """
        获取扩张边缘部分的mask（扩张区域 - 原始区域）
        
        Args:
            original_mask: 原始mask (bool array)
            kernel_size: 扩张核大小
            iterations: 迭代次数
            
        Returns:
            edge_mask: 边缘部分的mask
        """
        dilated_mask = self.dilate_mask(original_mask, kernel_size, iterations)
        edge_mask = dilated_mask & (~original_mask)
        return edge_mask
    
    def adjust_dilation_size(self, change):
        """调整扩张核大小"""
        self.dilation_kernel_size = max(1, self.dilation_kernel_size + change)
        print(f"📏 扩张核大小: {self.dilation_kernel_size}")
        
    def mouse_click(self, event, x, y, flags, param):
        """鼠标点击回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x, orig_y = self.display_coord_to_original(x, y)
            self.points.append([orig_x, orig_y])
            self.labels.append(1)
            print(f"添加正向点: ({orig_x}, {orig_y})")
            self.update_mask()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            orig_x, orig_y = self.display_coord_to_original(x, y)
            self.points.append([orig_x, orig_y])
            self.labels.append(0)
            print(f"添加负向点: ({orig_x}, {orig_y})")
            self.update_mask()
    
    def update_mask(self):
        """更新并显示mask"""
        if len(self.points) == 0:
            return
            
        points = np.array(self.points)
        labels = np.array(self.labels)
        
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        
        # 选择得分最高的mask
        best_mask = masks[np.argmax(scores)]
        self.current_mask = best_mask
        
        # 可视化显示
        if self.scale_factor != 1.0:
            display_mask = cv2.resize(
                best_mask.astype(np.uint8), 
                (self.display_image.shape[1], self.display_image.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            display_mask = best_mask
        
        display_image = self.display_image.copy()
        
        # 显示原始mask（绿色半透明）
        mask_overlay = display_image.copy()
        mask_overlay[display_mask] = [0, 255, 0]
        display_image = cv2.addWeighted(display_image, 0.7, mask_overlay, 0.3, 0)
        
        # 显示扩张后的mask（红色边框）
        dilated_mask = self.dilate_mask(best_mask)
        if self.scale_factor != 1.0:
            dilated_display = cv2.resize(
                dilated_mask.astype(np.uint8), 
                (self.display_image.shape[1], self.display_image.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            dilated_display = dilated_mask
            
        # 只显示扩张边缘（红色）
        edge_mask = dilated_display & (~display_mask)
        edge_overlay = display_image.copy()
        edge_overlay[edge_mask] = [0, 0, 255]
        display_image = cv2.addWeighted(display_image, 0.8, edge_overlay, 0.2, 0)
        
        # 显示点击点
        for point, label in zip(self.points, self.labels):
            disp_x, disp_y = self.original_coord_to_display(point[0], point[1])
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display_image, (disp_x, disp_y), 5, color, -1)
            cv2.circle(display_image, (disp_x, disp_y), 5, (255, 255, 255), 2)
        
        cv2.imshow('SAM Object Selector', display_image)
        
        # 显示选择区域统计
        area = np.sum(best_mask)
        dilated_area = np.sum(dilated_mask)
        edge_area = np.sum(self.get_edge_mask(best_mask))
        total_area = best_mask.shape[0] * best_mask.shape[1]
        percentage = (area / total_area) * 100
        dilated_percentage = (dilated_area / total_area) * 100
        edge_percentage = (edge_area / total_area) * 100
        print(f"原始区域: {area} 像素 ({percentage:.1f}%)")
        print(f"扩张区域: {dilated_area} 像素 ({dilated_percentage:.1f}%)")
        print(f"边缘区域: {edge_area} 像素 ({edge_percentage:.1f}%)")
        print(f"扩张核大小: {self.dilation_kernel_size}")
    
    def save_transparent_object(self, base_name):
        """保存透明背景的选中物体（使用原始精确mask）"""
        if self.current_mask is None:
            print("❌ 请先选择物体！")
            return None
        
        mask = self.current_mask
        image = self.original_image
        
        # 找到物体边界
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0:
            print("❌ 没有选中任何区域！")
            return None
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # 裁剪物体区域
        cropped_image = image[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        
        # 创建RGBA图像（透明背景）
        h, w = cropped_image.shape[:2]
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[:, :, :3] = cropped_image
        result[:, :, 3] = cropped_mask.astype(np.uint8) * 255
        
        # 保存到objects目录
        object_path = os.path.join(self.output_dirs['objects'], f"{base_name}_object.png")
        pil_image = Image.fromarray(result, 'RGBA')
        pil_image.save(object_path)
        
        print(f"✓ 透明背景物体: {object_path} ({w}x{h})")
        return object_path
    
    def save_background_with_hole(self, base_name, use_dilated_mask=True):
        """
        保存去掉物体后的背景（有洞）
        
        Args:
            base_name: 文件名前缀
            use_dilated_mask: 是否使用扩张后的mask
        """
        if self.current_mask is None:
            print("❌ 请先选择物体！")
            return None
        
        # 选择使用原始mask还是扩张mask
        if use_dilated_mask:
            mask = self.dilate_mask(self.current_mask)
            suffix = "_background_dilated_hole"
            print(f"🔍 使用扩张mask (核大小: {self.dilation_kernel_size})")
        else:
            mask = self.current_mask
            suffix = "_background_hole"
            print("🔍 使用原始精确mask")
        
        image = self.original_image
        
        # 创建有洞的背景
        background = image.copy()
        
        # 将选中区域设为黑色（表示洞）
        background[mask] = [0, 0, 0]
        
        # 保存到holes目录
        background_path = os.path.join(self.output_dirs['holes'], f"{base_name}{suffix}.jpg")
        pil_image = Image.fromarray(background)
        pil_image.save(background_path, quality=95)
        
        print(f"✓ 有洞背景: {background_path}")
        return background_path
    
    def save_mask_file(self, base_name, use_dilated_mask=True):
        """
        保存mask文件（给PixelHacker使用）
        
        Args:
            base_name: 文件名前缀
            use_dilated_mask: 是否使用扩张后的mask
        """
        if self.current_mask is None:
            print("❌ 请先选择物体！")
            return None
        
        # 选择使用原始mask还是扩张mask，并确定保存目录
        if use_dilated_mask:
            mask = self.dilate_mask(self.current_mask)
            suffix = "_mask"
            output_dir = self.output_dirs['mask_dilated']
            print(f"🔍 保存扩张mask (核大小: {self.dilation_kernel_size})")
        else:
            mask = self.current_mask
            suffix = "_mask"
            output_dir = self.output_dirs['mask_precise']
            print("🔍 保存原始精确mask")
        
        # 保存mask为PNG
        mask_uint8 = mask.astype(np.uint8) * 255
        mask_path = os.path.join(output_dir, f"{base_name}{suffix}.png")
        cv2.imwrite(mask_path, mask_uint8)
        
        print(f"✓ Mask文件: {mask_path}")
        return mask_path
    
    def save_edge_mask_file(self, base_name):
        """
        保存扩张边缘mask文件（仅包含扩张部分）
        
        Args:
            base_name: 文件名前缀
        """
        if self.current_mask is None:
            print("❌ 请先选择物体！")
            return None
        
        # 获取边缘mask
        edge_mask = self.get_edge_mask(self.current_mask)
        
        # 保存边缘mask为PNG
        mask_uint8 = edge_mask.astype(np.uint8) * 255
        edge_path = os.path.join(self.output_dirs['mask_edge'], f"{base_name}_edge_mask.png")
        cv2.imwrite(edge_path, mask_uint8)
        
        edge_area = np.sum(edge_mask)
        total_area = edge_mask.shape[0] * edge_mask.shape[1]
        edge_percentage = (edge_area / total_area) * 100
        
        print(f"✓ 边缘Mask文件: {edge_path}")
        print(f"🔍 边缘区域: {edge_area} 像素 ({edge_percentage:.1f}%) (核大小: {self.dilation_kernel_size})")
        return edge_path
    
    def save_all_outputs(self, base_name, use_dilated_for_inpainting=True):
        """
        保存所有输出文件到对应目录
        
        Args:
            base_name: 文件名前缀
            use_dilated_for_inpainting: inpainting相关文件是否使用扩张mask
        """
        if self.current_mask is None:
            print("❌ 请先选择物体！")
            return None
        
        print(f"\n保存所有输出文件到 {self.output_base_dir}")
        print("-" * 60)
        
        # 保存透明背景物体到objects目录（始终使用精确mask）
        object_path = self.save_transparent_object(base_name)
        
        # 保存背景到holes目录（可选择是否使用扩张mask）
        background_path = self.save_background_with_hole(base_name, use_dilated_for_inpainting)
        
        # 保存mask文件到对应目录
        if use_dilated_for_inpainting:
            mask_path = self.save_mask_file(base_name, True)  # 扩张mask到mask_dilated目录
        else:
            mask_path = self.save_mask_file(base_name, False)  # 精确mask到mask_precise目录
        
        # 同时保存另一种mask以备后用
        if use_dilated_for_inpainting:
            precise_mask_path = self.save_mask_file(base_name, False)  # 精确mask
        else:
            precise_mask_path = self.save_mask_file(base_name, True)   # 扩张mask
        
        # 保存边缘mask
        edge_mask_path = self.save_edge_mask_file(base_name)
        
        print("-" * 60)
        print("✅ 所有文件已保存到对应目录！")
        print(f"📁 输出目录结构:")
        print(f"   {self.output_dirs['objects']}")
        print(f"   {self.output_dirs['holes']}")
        print(f"   {self.output_dirs['mask_dilated']}")
        print(f"   {self.output_dirs['mask_precise']}")
        print(f"   {self.output_dirs['mask_edge']}")
        
        return {
            'object': object_path,
            'background_hole': background_path,
            'mask': mask_path,
            'alternative_mask': precise_mask_path,
            'edge_mask': edge_mask_path
        }
    
    def process_image(self, image_path):
        """处理图像"""
        # 读取图像
        self.original_image = cv2.imread(image_path)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        print(f"✓ 图像加载: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
        
        # 设置SAM图像
        self.predictor.set_image(self.original_image)
        
        # 准备显示
        display_rgb = self.resize_for_display(self.original_image)
        self.display_image = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
        
        # 创建窗口
        cv2.namedWindow('SAM Object Selector', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('SAM Object Selector', self.mouse_click)
        cv2.imshow('SAM Object Selector', self.display_image)
        
        print("\n" + "="*70)
        print("🎯 SAM 物体选择器 (支持Mask扩张优化)")
        print("="*70)
        print("📁 输出目录配置:")
        print(f"   物体文件: {self.output_dirs['objects']}")
        print(f"   背景文件: {self.output_dirs['holes']}")
        print(f"   扩张Mask: {self.output_dirs['mask_dilated']}")
        print(f"   精确Mask: {self.output_dirs['mask_precise']}")
        print(f"   边缘Mask: {self.output_dirs['mask_edge']}")
        print()
        print("🖱️  鼠标操作:")
        print("   左键: 选择物体区域")
        print("   右键: 排除区域")
        print("\n⌨️  键盘操作:")
        print("   'SPACE': 💾 保存所有输出 (默认用扩张mask)")
        print("   'ENTER': 💾 保存所有输出 (用精确mask)")
        print("   'o':     💾 只保存透明背景物体")
        print("   'b':     💾 保存背景 (扩张mask)")
        print("   'B':     💾 保存背景 (精确mask)")
        print("   'e':     💾 只保存边缘mask")
        print("   '+':     🔍 增大扩张核 (+2)")
        print("   '-':     🔍 减小扩张核 (-2)")
        print("   'r':     🔄 重置选择")
        print("   'q':     ❌ 退出")
        print("="*70)
        print("💡 绿色=选中区域, 红色边缘=扩张区域")
        print("💡 推荐：用扩张mask进行inpainting可获得更好效果！")
        print("="*70)
        
        # 主循环
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 退出程序")
                break
            elif key == ord(' '):  # 空格键 - 保存所有输出（扩张mask）
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                # 检查是否按了Shift
                if cv2.getWindowProperty('SAM Object Selector', cv2.WND_PROP_VISIBLE) >= 1:
                    results = self.save_all_outputs(base_name, use_dilated_for_inpainting=True)
                if results:
                    print(f"\n🎉 输出文件已保存 (使用扩张mask):")
                    print(f"   📦 物体文件: {os.path.basename(results['object'])}")
                    print(f"   🕳️  背景文件: {os.path.basename(results['background_hole'])}")
                    print(f"   🎭 主要Mask: {os.path.basename(results['mask'])}")
                    print(f"   🎭 备用Mask: {os.path.basename(results['alternative_mask'])}")
                    print(f"   🔲 边缘Mask: {os.path.basename(results['edge_mask'])}")
                    print(f"\n📁 文件位置:")
                    print(f"   {self.output_base_dir}/")
                    print(f"   ├── objects/")
                    print(f"   ├── holes/")
                    print(f"   ├── mask_dilated/")
                    print(f"   ├── mask_precise/")
                    print(f"   └── mask_edge/")
                    print(f"\n💡 下一步: 使用PixelHacker修复背景")
                    print(f"   推荐使用holes/目录中的扩张文件获得更好效果！")
            elif key == 13:  # Enter键 - 保存所有输出（精确mask）
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                results = self.save_all_outputs(base_name, use_dilated_for_inpainting=False)
                if results:
                    print(f"\n🎉 输出文件已保存 (使用精确mask):")
                    print(f"   📦 物体文件: {os.path.basename(results['object'])}")
                    print(f"   🕳️  背景文件: {os.path.basename(results['background_hole'])}")
                    print(f"   🎭 主要Mask: {os.path.basename(results['mask'])}")
                    print(f"   🎭 备用Mask: {os.path.basename(results['alternative_mask'])}")
                    print(f"   🔲 边缘Mask: {os.path.basename(results['edge_mask'])}")
            elif key == ord('o'):
                # 只保存物体
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                self.save_transparent_object(base_name)
            elif key == ord('b'):
                # 保存背景（扩张mask）
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                self.save_background_with_hole(base_name, use_dilated_mask=True)
            elif key == ord('B'):
                # 保存背景（精确mask）
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                self.save_background_with_hole(base_name, use_dilated_mask=False)
            elif key == ord('e'):
                # 只保存边缘mask
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                self.save_edge_mask_file(base_name)
            elif key == ord('+') or key == ord('='):
                # 增大扩张核
                self.adjust_dilation_size(2)
                if len(self.points) > 0:
                    self.update_mask()
            elif key == ord('-'):
                # 减小扩张核
                self.adjust_dilation_size(-2)
                if len(self.points) > 0:
                    self.update_mask()
            elif key == ord('r'):
                # 重置选择
                self.points = []
                self.labels = []
                self.current_mask = None
                cv2.imshow('SAM Object Selector', self.display_image)
                print("🔄 已重置选择")
        
        cv2.destroyAllWindows()

def main():
    # 查找图像文件
    image_path = "/home/liushuzhi/pax/2.5d_editing/car.jpg"
    
    if not os.path.exists(image_path):
        # 自动查找图像文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        current_images = [f for f in os.listdir('.') if f.lower().endswith(image_extensions)]
        
        if current_images:
            image_path = current_images[0]
            print(f"📸 自动选择图像: {image_path}")
        else:
            print("❌ 未找到图像文件！")
            print("请将图像文件放在当前目录下")
            return
    
    # 创建处理器并运行
    processor = SimplifiedSAMProcessor(max_display_size=800)
    processor.process_image(image_path)

if __name__ == "__main__":
    ensure_sam_checkpoint()
    main()