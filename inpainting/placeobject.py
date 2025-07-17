import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import math

class AdvancedImageOverlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("拖拽缩放图像合成器")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 初始化变量
        self.background_image = None
        self.overlay_image = None
        self.background_path = None
        self.overlay_path = None
        self.result_image = None
        
        # 前景图片变换参数
        self.overlay_x = 0
        self.overlay_y = 0
        self.overlay_scale = 1.0
        self.overlay_rotation = 0
        self.overlay_opacity = 1.0
        
        # 画布和预览相关
        self.canvas_scale = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        
        # 拖拽相关
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.mouse_over_overlay = False
        
        # 渲染缓存
        self.background_photo = None
        self.overlay_photo = None
        self.composite_photo = None
        
        # 创建UI
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="拖拽缩放图像合成器", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 左侧控制面板
        control_panel = ttk.Frame(main_frame, width=300)
        control_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        control_panel.grid_propagate(False)
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(control_panel, text="图片选择", padding="15")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        file_frame.columnconfigure(0, weight=1)
        
        ttk.Button(file_frame, text="选择背景图片", 
                  command=self.select_background).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.bg_path_label = ttk.Label(file_frame, text="未选择文件", foreground="gray", wraplength=250)
        self.bg_path_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="选择前景图片", 
                  command=self.select_overlay).grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.fg_path_label = ttk.Label(file_frame, text="未选择文件", foreground="gray", wraplength=250)
        self.fg_path_label.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # 精确控制区域
        precision_frame = ttk.LabelFrame(control_panel, text="精确控制", padding="15")
        precision_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # 位置输入
        pos_frame = ttk.Frame(precision_frame)
        pos_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        pos_frame.columnconfigure(1, weight=1)
        pos_frame.columnconfigure(3, weight=1)
        
        ttk.Label(pos_frame, text="X:").grid(row=0, column=0, padx=(0, 5))
        self.x_var = tk.IntVar(value=0)
        self.x_entry = ttk.Entry(pos_frame, textvariable=self.x_var, width=8)
        self.x_entry.grid(row=0, column=1, padx=(0, 15))
        self.x_entry.bind('<Return>', self.on_manual_input)
        
        ttk.Label(pos_frame, text="Y:").grid(row=0, column=2, padx=(0, 5))
        self.y_var = tk.IntVar(value=0)
        self.y_entry = ttk.Entry(pos_frame, textvariable=self.y_var, width=8)
        self.y_entry.grid(row=0, column=3)
        self.y_entry.bind('<Return>', self.on_manual_input)
        
        # 缩放控制
        scale_frame = ttk.Frame(precision_frame)
        scale_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        scale_frame.columnconfigure(1, weight=1)
        
        ttk.Label(scale_frame, text="缩放:").grid(row=0, column=0, padx=(0, 10))
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_scale = ttk.Scale(scale_frame, from_=0.1, to=5.0, 
                                    variable=self.scale_var, orient=tk.HORIZONTAL,
                                    command=self.on_scale_change)
        self.scale_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.scale_label = ttk.Label(scale_frame, text="100%")
        self.scale_label.grid(row=0, column=2)
        
        # 旋转控制
        rotation_frame = ttk.Frame(precision_frame)
        rotation_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        rotation_frame.columnconfigure(1, weight=1)
        
        ttk.Label(rotation_frame, text="旋转:").grid(row=0, column=0, padx=(0, 10))
        self.rotation_var = tk.IntVar(value=0)
        self.rotation_scale = ttk.Scale(rotation_frame, from_=-180, to=180, 
                                       variable=self.rotation_var, orient=tk.HORIZONTAL,
                                       command=self.on_rotation_change)
        self.rotation_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.rotation_label = ttk.Label(rotation_frame, text="0°")
        self.rotation_label.grid(row=0, column=2)
        
        # 透明度控制
        opacity_frame = ttk.Frame(precision_frame)
        opacity_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        opacity_frame.columnconfigure(1, weight=1)
        
        ttk.Label(opacity_frame, text="透明度:").grid(row=0, column=0, padx=(0, 10))
        self.opacity_var = tk.DoubleVar(value=1.0)
        self.opacity_scale = ttk.Scale(opacity_frame, from_=0.0, to=1.0, 
                                      variable=self.opacity_var, orient=tk.HORIZONTAL,
                                      command=self.on_opacity_change)
        self.opacity_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.opacity_label = ttk.Label(opacity_frame, text="100%")
        self.opacity_label.grid(row=0, column=2)
        
        # 操作说明
        tips_frame = ttk.LabelFrame(control_panel, text="操作说明", padding="15")
        tips_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        tips_text = """🖱️ 鼠标操作：
• 左键拖拽前景图片移动位置
• 滚轮放大/缩小前景图片
• 右键重置前景图片到中心

⌨️ 键盘操作：
• 在输入框中输入精确坐标
• 使用滑块微调各种参数

💡 提示：
• 前景图片可以超出背景边界
• 支持透明PNG图片效果最佳"""
        
        tips_label = ttk.Label(tips_frame, text=tips_text, font=('Arial', 9), justify=tk.LEFT)
        tips_label.grid(row=0, column=0, sticky=tk.W)
        
        # 按钮区域
        button_frame = ttk.Frame(control_panel)
        button_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        button_frame.columnconfigure(0, weight=1)
        
        ttk.Button(button_frame, text="🎨 生成合成图片", 
                  command=self.generate_result).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        ttk.Button(button_frame, text="💾 保存图片", 
                  command=self.save_result).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        ttk.Button(button_frame, text="🔄 重置", 
                  command=self.reset).grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        # 右侧预览区域
        preview_frame = ttk.LabelFrame(main_frame, text="实时预览 - 鼠标拖拽 + 滚轮缩放", padding="10")
        preview_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # 创建画布
        self.canvas = tk.Canvas(preview_frame, bg='#f8f8f8', cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<MouseWheel>", self.on_canvas_scroll)
        self.canvas.bind("<Button-4>", self.on_canvas_scroll)  # Linux
        self.canvas.bind("<Button-5>", self.on_canvas_scroll)  # Linux
        
        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(status_frame, text="请选择背景图片和前景图片开始合成", 
                                     font=('Arial', 10), foreground='gray')
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        main_frame.rowconfigure(1, weight=1)
        
    def select_background(self):
        """选择背景图片"""
        file_path = filedialog.askopenfilename(
            title="选择背景图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            try:
                self.background_image = Image.open(file_path)
                self.background_path = file_path
                filename = os.path.basename(file_path)
                self.bg_path_label.config(text=f"✅ {filename}", foreground="darkgreen")
                self.status_label.config(text=f"背景图片已加载: {filename}")
                self.update_canvas()
            except Exception as e:
                messagebox.showerror("错误", f"无法打开背景图片: {str(e)}")
    
    def select_overlay(self):
        """选择前景图片"""
        file_path = filedialog.askopenfilename(
            title="选择前景图片（推荐PNG透明格式）",
            filetypes=[("PNG图片", "*.png"), ("所有图片", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            try:
                self.overlay_image = Image.open(file_path)
                if self.overlay_image.mode != 'RGBA':
                    self.overlay_image = self.overlay_image.convert('RGBA')
                self.overlay_path = file_path
                filename = os.path.basename(file_path)
                self.fg_path_label.config(text=f"✅ {filename}", foreground="darkgreen")
                self.status_label.config(text=f"前景图片已加载: {filename}")
                
                # 初始化前景图片位置（居中）
                if self.background_image:
                    self.overlay_x = (self.background_image.width - self.overlay_image.width) // 2
                    self.overlay_y = (self.background_image.height - self.overlay_image.height) // 2
                    self.update_ui_values()
                
                self.update_canvas()
            except Exception as e:
                messagebox.showerror("错误", f"无法打开前景图片: {str(e)}")
    
    def get_overlay_bounds(self):
        """获取前景图片的边界"""
        if not self.overlay_image:
            return None
            
        # 计算缩放后的尺寸
        scaled_width = int(self.overlay_image.width * self.overlay_scale)
        scaled_height = int(self.overlay_image.height * self.overlay_scale)
        
        # 考虑旋转后的边界（简化处理）
        if self.overlay_rotation != 0:
            # 旋转后的边界可能会更大，这里做简单估算
            diagonal = math.sqrt(scaled_width**2 + scaled_height**2)
            scaled_width = scaled_height = int(diagonal)
        
        return {
            'x': self.overlay_x - scaled_width // 2,
            'y': self.overlay_y - scaled_height // 2,
            'width': scaled_width,
            'height': scaled_height
        }
    
    def point_in_overlay(self, canvas_x, canvas_y):
        """检查点是否在前景图片内"""
        bounds = self.get_overlay_bounds()
        if not bounds:
            return False
            
        # 转换画布坐标到图片坐标
        img_x = (canvas_x / self.canvas_scale) - self.canvas_offset_x
        img_y = (canvas_y / self.canvas_scale) - self.canvas_offset_y
        
        return (bounds['x'] <= img_x <= bounds['x'] + bounds['width'] and
                bounds['y'] <= img_y <= bounds['y'] + bounds['height'])
    
    def on_canvas_motion(self, event):
        """鼠标移动事件"""
        if self.point_in_overlay(event.x, event.y):
            if not self.mouse_over_overlay:
                self.canvas.config(cursor="hand2")
                self.mouse_over_overlay = True
        else:
            if self.mouse_over_overlay:
                self.canvas.config(cursor="crosshair")
                self.mouse_over_overlay = False
    
    def on_canvas_click(self, event):
        """画布点击事件"""
        if not self.background_image or not self.overlay_image:
            return
        
        if self.point_in_overlay(event.x, event.y):
            self.dragging = True
            # 记录拖拽起始点
            img_x = (event.x / self.canvas_scale) - self.canvas_offset_x
            img_y = (event.y / self.canvas_scale) - self.canvas_offset_y
            self.drag_start_x = img_x - self.overlay_x
            self.drag_start_y = img_y - self.overlay_y
            self.canvas.config(cursor="move")
    
    def on_canvas_drag(self, event):
        """画布拖拽事件"""
        if not self.dragging:
            return
        
        # 转换画布坐标到图片坐标
        img_x = (event.x / self.canvas_scale) - self.canvas_offset_x
        img_y = (event.y / self.canvas_scale) - self.canvas_offset_y
        
        # 更新前景图片位置
        self.overlay_x = img_x - self.drag_start_x
        self.overlay_y = img_y - self.drag_start_y
        
        # 更新UI显示
        self.update_ui_values()
        self.update_canvas()
        
        # 更新状态
        self.status_label.config(text=f"位置: ({int(self.overlay_x)}, {int(self.overlay_y)})")
    
    def on_canvas_release(self, event):
        """画布释放事件"""
        self.dragging = False
        if self.mouse_over_overlay:
            self.canvas.config(cursor="hand2")
        else:
            self.canvas.config(cursor="crosshair")
    
    def on_canvas_right_click(self, event):
        """右键重置位置"""
        if self.background_image and self.overlay_image:
            # 重置到中心
            self.overlay_x = self.background_image.width // 2
            self.overlay_y = self.background_image.height // 2
            self.update_ui_values()
            self.update_canvas()
            self.status_label.config(text="前景图片已重置到中心位置")
    
    def on_canvas_scroll(self, event):
        """滚轮缩放事件"""
        if not self.background_image or not self.overlay_image:
            return
        
        # 检查鼠标是否在前景图片上
        if not self.point_in_overlay(event.x, event.y):
            return
        
        # 计算缩放因子
        if event.delta > 0 or event.num == 4:  # 向上滚动
            scale_factor = 1.1
        else:  # 向下滚动
            scale_factor = 0.9
        
        # 应用缩放
        new_scale = self.overlay_scale * scale_factor
        new_scale = max(0.1, min(5.0, new_scale))  # 限制范围
        
        if new_scale != self.overlay_scale:
            self.overlay_scale = new_scale
            self.update_ui_values()
            self.update_canvas()
            self.status_label.config(text=f"缩放: {int(new_scale * 100)}%")
    
    def on_manual_input(self, event):
        """手动输入坐标"""
        try:
            self.overlay_x = self.x_var.get()
            self.overlay_y = self.y_var.get()
            self.update_canvas()
        except:
            pass
    
    def on_scale_change(self, value):
        """滑块缩放改变"""
        self.overlay_scale = float(value)
        self.scale_label.config(text=f"{int(self.overlay_scale * 100)}%")
        self.update_canvas()
    
    def on_rotation_change(self, value):
        """旋转改变"""
        self.overlay_rotation = int(float(value))
        self.rotation_label.config(text=f"{self.overlay_rotation}°")
        self.update_canvas()
    
    def on_opacity_change(self, value):
        """透明度改变"""
        self.overlay_opacity = float(value)
        self.opacity_label.config(text=f"{int(self.overlay_opacity * 100)}%")
        self.update_canvas()
    
    def update_ui_values(self):
        """更新UI控件的值"""
        self.x_var.set(int(self.overlay_x))
        self.y_var.set(int(self.overlay_y))
        self.scale_var.set(self.overlay_scale)
        self.scale_label.config(text=f"{int(self.overlay_scale * 100)}%")
    
    def update_canvas(self):
        """更新画布显示"""
        if not self.background_image:
            return
        
        # 清空画布
        self.canvas.delete("all")
        
        # 获取画布尺寸
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width, canvas_height = 800, 600
        
        # 计算缩放比例以适应画布
        img_width, img_height = self.background_image.size
        self.canvas_scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        
        # 计算居中偏移
        scaled_width = int(img_width * self.canvas_scale)
        scaled_height = int(img_height * self.canvas_scale)
        self.canvas_offset_x = (canvas_width - scaled_width) // 2
        self.canvas_offset_y = (canvas_height - scaled_height) // 2
        
        # 绘制背景图片
        if self.canvas_scale < 1.0:
            display_bg = self.background_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        else:
            display_bg = self.background_image
        
        self.background_photo = ImageTk.PhotoImage(display_bg)
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, 
                                anchor=tk.NW, image=self.background_photo)
        
        # 绘制前景图片
        if self.overlay_image:
            composite = self.generate_composite_image()
            if composite:
                if self.canvas_scale < 1.0:
                    display_composite = composite.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                else:
                    display_composite = composite
                
                self.composite_photo = ImageTk.PhotoImage(display_composite)
                self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, 
                                        anchor=tk.NW, image=self.composite_photo)
        
        # 绘制前景图片边界框（调试用）
        if self.overlay_image:
            bounds = self.get_overlay_bounds()
            if bounds:
                x1 = bounds['x'] * self.canvas_scale + self.canvas_offset_x
                y1 = bounds['y'] * self.canvas_scale + self.canvas_offset_y
                x2 = x1 + bounds['width'] * self.canvas_scale
                y2 = y1 + bounds['height'] * self.canvas_scale
                
                # 绘制半透明边界框
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2, dash=(5, 5))
    
    def generate_composite_image(self):
        """生成合成图像"""
        if not self.background_image or not self.overlay_image:
            return None
        
        # 复制背景图像
        result = self.background_image.copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')
        
        # 处理前景图像
        overlay = self.overlay_image.copy()
        
        # 缩放
        if self.overlay_scale != 1.0:
            new_width = int(overlay.width * self.overlay_scale)
            new_height = int(overlay.height * self.overlay_scale)
            overlay = overlay.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 旋转
        if self.overlay_rotation != 0:
            overlay = overlay.rotate(self.overlay_rotation, expand=True)
        
        # 透明度
        if self.overlay_opacity < 1.0:
            alpha = overlay.split()[-1]
            alpha = alpha.point(lambda p: int(p * self.overlay_opacity))
            overlay.putalpha(alpha)
        
        # 计算粘贴位置（以中心为基准）
        paste_x = int(self.overlay_x - overlay.width // 2)
        paste_y = int(self.overlay_y - overlay.height // 2)
        
        # 合成
        result.paste(overlay, (paste_x, paste_y), overlay)
        
        return result
    
    def generate_result(self):
        """生成最终结果"""
        if not self.background_image or not self.overlay_image:
            messagebox.showwarning("警告", "请先选择背景图片和前景图片")
            return
        
        try:
            self.result_image = self.generate_composite_image()
            messagebox.showinfo("成功", "✅ 合成图片生成成功！可以点击'保存图片'按钮保存结果。")
            self.status_label.config(text="合成图片已生成，可以保存")
        except Exception as e:
            messagebox.showerror("错误", f"生成合成图片时出错: {str(e)}")
    
    def save_result(self):
        """保存结果图片"""
        if not self.result_image:
            messagebox.showwarning("警告", "请先生成合成图片")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存合成图片",
            defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("JPEG图片", "*.jpg"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    # JPEG格式需要转换为RGB
                    rgb_image = Image.new('RGB', self.result_image.size, (255, 255, 255))
                    rgb_image.paste(self.result_image, mask=self.result_image.split()[-1] if self.result_image.mode == 'RGBA' else None)
                    rgb_image.save(file_path, quality=95)
                else:
                    self.result_image.save(file_path)
                
                messagebox.showinfo("成功", f"✅ 图片已保存到:\n{file_path}")
                self.status_label.config(text=f"图片已保存: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"保存图片时出错: {str(e)}")
    
    def reset(self):
        """重置所有设置"""
        self.background_image = None
        self.overlay_image = None
        self.background_path = None
        self.overlay_path = None
        self.result_image = None
        
        self.overlay_x = 0
        self.overlay_y = 0
        self.overlay_scale = 1.0
        self.overlay_rotation = 0
        self.overlay_opacity = 1.0
        
        self.bg_path_label.config(text="未选择文件", foreground="gray")
        self.fg_path_label.config(text="未选择文件", foreground="gray")
        self.status_label.config(text="已重置，请重新选择图片")
        
        self.update_ui_values()
        self.canvas.delete("all")

def main():
    root = tk.Tk()
    app = AdvancedImageOverlayApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()