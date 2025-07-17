import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

class StableDiffusionUI:
    def __init__(self):
        print("正在加载Stable Diffusion模型...")
        
        # 文本到图像模型选择（按推荐程度排序）
        text2img_models = [
            "SG161222/Realistic_Vision_V5.1_noVAE",  # 推荐：真实感照片风格，质量很高
            "Lykon/DreamShaper",                     # 推荐：通用高质量模型
            # "stabilityai/stable-diffusion-2-1",   # 备选：官方改进版
            # "andite/anything-v4.0",               # 备选：动漫风格
            "runwayml/stable-diffusion-v1-5",       # 原始模型作为最后备选
        ]
        
        # 尝试加载文本到图像模型
        self.text2img_pipe = None
        for model_id in text2img_models:
            try:
                print(f"尝试加载模型: {model_id}")
                self.text2img_pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    # safety_checker=None,  # 可选：禁用安全检查器节省显存
                    # requires_safety_checker=False
                ).to("cuda")
                self.text2img_pipe.enable_attention_slicing()
                self.text2img_pipe.enable_xformers_memory_efficient_attention()
                print(f"成功加载模型: {model_id}")
                break
            except Exception as e:
                print(f"模型 {model_id} 加载失败: {e}")
                continue
        
        if self.text2img_pipe is None:
            raise Exception("所有文本到图像模型都加载失败")
        
        # Inpainting模型 - 保留原有逻辑
        print("正在加载Inpainting模型...")
        self.inpaint_pipe = None
        
        try:
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
            ).to("cuda")
            print("Inpainting模型加载成功!")
        except Exception as e:
            print(f"主模型加载失败: {e}")
            try:
                print("尝试使用备用inpainting模型...")
                self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    torch_dtype=torch.float16,
                ).to("cuda")
                print("备用Inpainting模型加载成功!")
            except Exception as e2:
                print(f"所有inpainting模型加载失败: {e2}")
        
        if self.inpaint_pipe:
            self.inpaint_pipe.enable_attention_slicing()
            self.inpaint_pipe.enable_xformers_memory_efficient_attention()
        
        print("模型初始化完成!")

    def generate_image(self, prompt, negative_prompt, num_steps, guidance_scale, width, height):
        """生成文本到图像"""
        try:
            if not prompt.strip():
                return None, "请输入提示词"
            
            # 可选：简单的提示词增强
            # if "highly detailed" not in prompt.lower() and "detailed" not in prompt.lower():
            #     prompt = f"{prompt}, highly detailed, best quality"
            
            image = self.text2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]
            
            return image, "图像生成成功!"
        except Exception as e:
            return None, f"生成失败: {str(e)}"

    def process_editor_output(self, editor_output):
        """简化的编辑器输出处理"""
        if editor_output is None:
            return None, None
        
        # 处理字典格式输出
        if isinstance(editor_output, dict):
            if 'composite' in editor_output:
                composite = editor_output['composite']
                background = editor_output.get('background', composite)
                
                # 从composite创建mask
                if isinstance(composite, Image.Image):
                    img_array = np.array(composite)
                    # 检测白色绘制区域
                    if len(img_array.shape) == 3:
                        mask_array = np.all(img_array > 250, axis=2)
                    else:
                        mask_array = img_array > 250
                    mask = Image.fromarray((mask_array * 255).astype(np.uint8))
                    return background, mask
            
            # 直接的image/mask格式
            elif 'image' in editor_output:
                return editor_output.get('image'), editor_output.get('mask')
        
        return None, None

    def inpaint_image(self, original_img, mask_img, prompt, negative_prompt, num_steps, guidance_scale, strength):
        """执行inpainting"""
        try:
            if not self.inpaint_pipe:
                return None, "Inpainting模型未加载成功"
            
            if not all([original_img, mask_img, prompt.strip()]):
                return None, "请提供完整的输入（原图、mask、提示词）"
            
            # 强制resize到模型要求的尺寸
            target_size = (512, 512)
            if original_img.size != target_size:
                original_img = original_img.resize(target_size, Image.LANCZOS)
            if mask_img.size != target_size:
                mask_img = mask_img.resize(target_size, Image.NEAREST)

            # 确保 image 为 RGB
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')

            # 确保 mask 为单通道且二值化
            mask_img = mask_img.convert('L')
            mask_np = np.array(mask_img)
            mask_np = np.where(mask_np > 127, 255, 0).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)

            print("image size:", original_img.size, "mode:", original_img.mode)
            print("mask size:", mask_img.size, "mode:", mask_img.mode)
            print("mask unique values:", np.unique(mask_np))

            result = self.inpaint_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=original_img,
                mask_image=mask_img,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                strength=strength
            ).images[0]
            
            return result, "Inpainting完成!"
            
        except Exception as e:
            import traceback; traceback.print_exc()
            return None, f"Inpainting失败: {str(e)}"

    def inpaint_from_editor(self, editor_output, prompt, negative_prompt, num_steps, guidance_scale, strength):
        """从编辑器输出执行inpainting"""
        if editor_output is None:
            return None, "请先上传图像并绘制mask"
        
        original_img, mask_img = self.process_editor_output(editor_output)
        
        if original_img is None or mask_img is None:
            return None, "无法从编辑器提取图像和mask"
        
        return self.inpaint_image(original_img, mask_img, prompt, negative_prompt, num_steps, guidance_scale, strength)

def create_interface():
    sd_ui = StableDiffusionUI()
    inpainting_available = sd_ui.inpaint_pipe is not None
    
    with gr.Blocks(title="Stable Diffusion Studio", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 Stable Diffusion Inpainting Studio")
        gr.Markdown("生成图像并进行局部修改")
        
        if not inpainting_available:
            gr.Markdown("⚠️ **注意**: Inpainting模型加载失败，只能使用文本生成图像功能")
        
        with gr.Tabs():
            # 文本生成图像
            with gr.TabItem("📝 文本生成图像"):
                with gr.Row():
                    with gr.Column():
                        txt_prompt = gr.Textbox(
                            label="提示词", 
                            placeholder="描述你想要生成的图像...",
                            lines=3
                        )
                        txt_negative = gr.Textbox(
                            label="负面提示词", 
                            placeholder="不想要的元素...",
                            value="blurry, low quality, distorted"
                        )
                        
                        with gr.Row():
                            txt_steps = gr.Slider(10, 50, value=25, label="推理步数")  # 稍微调高默认值
                            txt_guidance = gr.Slider(1, 20, value=8.0, label="引导强度")  # 稍微调高
                        
                        with gr.Row():
                            txt_width = gr.Slider(256, 1024, value=512, step=64, label="宽度")
                            txt_height = gr.Slider(256, 1024, value=512, step=64, label="高度")
                        
                        txt_btn = gr.Button("🎨 生成图像", variant="primary")
                    
                    with gr.Column():
                        txt_output = gr.Image(label="生成的图像", type="pil")
                        txt_status = gr.Textbox(label="状态", interactive=False)
            
            # Inpainting功能
            if inpainting_available:
                with gr.TabItem("🖌️ 分离上传Inpainting"):
                    gr.Markdown("分别上传原图和mask图像")
                    
                    with gr.Row():
                        with gr.Column():
                            inpaint_original = gr.Image(label="原始图像", type="pil")
                            inpaint_mask = gr.Image(
                                label="Mask图像（白色=修改区域，黑色=保留区域）",
                                type="pil",
                                image_mode="L"
                            )
                            
                            inpaint_prompt = gr.Textbox(
                                label="Inpainting提示词",
                                placeholder="描述要在mask区域生成的内容...",
                                lines=2
                            )
                            inpaint_negative = gr.Textbox(
                                label="负面提示词",
                                value="blurry, low quality, distorted"
                            )
                            
                            with gr.Row():
                                inpaint_steps = gr.Slider(10, 50, value=25, label="推理步数")
                                inpaint_guidance = gr.Slider(1, 20, value=8.0, label="引导强度")
                                inpaint_strength = gr.Slider(0.1, 1.0, value=0.75, label="修改强度")
                            
                            inpaint_btn = gr.Button("🖌️ 执行Inpainting", variant="primary")
                        
                        with gr.Column():
                            inpaint_output = gr.Image(label="Inpainting结果")
                            inpaint_status = gr.Textbox(label="状态", interactive=False)
                
                with gr.TabItem("🖌️ 画板编辑Inpainting"):
                    gr.Markdown("直接在图像上绘制mask区域")
                    
                    with gr.Row():
                        with gr.Column():
                            editor = gr.ImageEditor(
                                label="上传图像并用白色画笔绘制要修改的区域",
                                type="pil",
                                brush=gr.Brush(
                                    colors=["#FFFFFF"],
                                    default_color="#FFFFFF",
                                    color_mode="fixed",
                                    default_size=20
                                ),
                                height=400
                            )
                            
                            editor_prompt = gr.Textbox(
                                label="Inpainting提示词",
                                placeholder="描述要在mask区域生成的内容...",
                                lines=2
                            )
                            editor_negative = gr.Textbox(
                                label="负面提示词",
                                value="blurry, low quality, distorted"
                            )
                            
                            with gr.Row():
                                editor_steps = gr.Slider(10, 50, value=25, label="推理步数")
                                editor_guidance = gr.Slider(1, 20, value=8.0, label="引导强度")
                                editor_strength = gr.Slider(0.1, 1.0, value=0.75, label="修改强度")
                            
                            editor_btn = gr.Button("🖌️ 执行Inpainting", variant="primary")
                        
                        with gr.Column():
                            editor_output = gr.Image(label="Inpainting结果")
                            editor_status = gr.Textbox(label="状态", interactive=False)
        
        # 传递按钮
        if inpainting_available:
            with gr.Row():
                gr.Markdown("### 快速传递")
                transfer_btn1 = gr.Button("📤 传递到分离上传", variant="secondary")
                transfer_btn2 = gr.Button("📤 传递到画板编辑", variant="secondary")
        
        # 示例 - 优化了提示词
        gr.Examples(
            examples=[
                ["portrait of a beautiful woman, professional photography, highly detailed", "blurry, low quality"],
                ["a cute cat sitting on a windowsill, soft lighting, detailed", "blurry, distorted"],
                ["cyberpunk city at night, neon lights, futuristic, detailed", "low quality, blurry"],
                ["mountain landscape, golden hour, highly detailed, professional photography", "blurry, low quality"],
            ],
            inputs=[txt_prompt, txt_negative]
        )
        
        # 事件绑定
        txt_btn.click(
            sd_ui.generate_image,
            inputs=[txt_prompt, txt_negative, txt_steps, txt_guidance, txt_width, txt_height],
            outputs=[txt_output, txt_status]
        )
        
        if inpainting_available:
            inpaint_btn.click(
                sd_ui.inpaint_image,
                inputs=[inpaint_original, inpaint_mask, inpaint_prompt, inpaint_negative, 
                       inpaint_steps, inpaint_guidance, inpaint_strength],
                outputs=[inpaint_output, inpaint_status]
            )
            
            editor_btn.click(
                sd_ui.inpaint_from_editor,
                inputs=[editor, editor_prompt, editor_negative,
                       editor_steps, editor_guidance, editor_strength],
                outputs=[editor_output, editor_status]
            )
            
            # 传递功能
            transfer_btn1.click(
                lambda img: img,
                inputs=[txt_output],
                outputs=[inpaint_original]
            )
            
            transfer_btn2.click(
                lambda img: {"background": img, "layers": [], "composite": img} if img else None,
                inputs=[txt_output],
                outputs=[editor]
            )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )