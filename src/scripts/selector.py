import gradio as gr
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

print(f"Gradio version: {gr.__version__}")

# --- 配置 ---
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')

class ImageSelector:
    """动漫转绘图片挑选器"""
    
    def __init__(self):
        self.indexed_data = {}
        self.original_keys = []
        self.current_index = 0
        self.selected_map = {}  # {base_name: [list of selected paths]}
        self.current_stylized_paths = []  # 当前显示的转绘图片路径
        
    def get_base_filename(self, filepath: str) -> str:
        """获取文件基础名称（不含扩展名）"""
        return Path(filepath).stem
    
    def find_images_recursive(self, folder_path: str) -> List[str]:
        """递归查找文件夹中的所有图片"""
        if not folder_path or not os.path.isdir(folder_path):
            return []
        
        images = []
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if fname.lower().endswith(ALLOWED_EXTENSIONS):
                    images.append(os.path.join(root, fname))
        
        return sorted(images)
    
    def load_and_index(self, original_folder: str, stylized_folder: str) -> Tuple:
        """加载并索引图片"""
        # 验证输入
        if not original_folder or not os.path.isdir(original_folder):
            return self._error_result(f"❌ 原图文件夹路径无效: {original_folder}")
        
        if not stylized_folder or not os.path.isdir(stylized_folder):
            return self._error_result(f"❌ 转绘文件夹路径无效: {stylized_folder}")
        
        # 查找图片
        original_images = self.find_images_recursive(original_folder)
        stylized_images = self.find_images_recursive(stylized_folder)
        
        if not original_images:
            return self._error_result("❌ 原图文件夹中未找到任何图片")
        
        if not stylized_images:
            return self._error_result("❌ 转绘文件夹中未找到任何图片")
        
        # 建立索引
        self.indexed_data = {}
        for orig_path in original_images:
            orig_base = self.get_base_filename(orig_path)
            matched_stylized = []
            
            for stylized_path in stylized_images:
                stylized_base = self.get_base_filename(stylized_path)
                if stylized_base.startswith(orig_base):
                    matched_stylized.append(stylized_path)
            
            if matched_stylized:
                self.indexed_data[orig_path] = {
                    "original_path": orig_path,
                    "stylized_paths": sorted(matched_stylized),
                    "base_name": orig_base
                }
        
        if not self.indexed_data:
            return self._error_result("❌ 未找到任何匹配的图片对，请检查文件名规则")
        
        # 初始化状态
        self.original_keys = sorted(list(self.indexed_data.keys()))
        self.current_index = 0
        self.selected_map = {data["base_name"]: [] for data in self.indexed_data.values()}
        
        # 返回首张图片的信息
        first_data = self.indexed_data[self.original_keys[0]]
        self.current_stylized_paths = first_data["stylized_paths"]
        status = f"✅ 加载成功！找到 {len(self.original_keys)} 组匹配的图片"
        nav_info = self._get_nav_info()
        selected_display = self._get_selected_display()
        
        # 生成复选框和图片显示
        checkboxes, images = self._generate_stylized_display(first_data["stylized_paths"], first_data["base_name"])
        
        return (
            status,
            first_data["original_path"],
            nav_info,
            selected_display,
            gr.update(interactive=True),  # prev_btn
            gr.update(interactive=True),  # next_btn
            gr.update(interactive=True),  # save_btn
            *checkboxes,  # 返回所有复选框状态
            *images      # 返回所有图片
        )
    
    def _error_result(self, message: str) -> Tuple:
        """返回错误结果"""
        empty_checkboxes = [gr.update(visible=False, value=False) for _ in range(6)]
        empty_images = [gr.update(visible=False, value=None) for _ in range(6)]
        
        return (
            message, None, "无数据", [],
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            *empty_checkboxes,
            *empty_images
        )
    
    def _get_nav_info(self) -> str:
        """获取导航信息"""
        if not self.original_keys:
            return "无数据"
        
        current_name = os.path.basename(self.original_keys[self.current_index])
        current_data = self.indexed_data[self.original_keys[self.current_index]]
        base_name = current_data["base_name"]
        selected_count = len(self.selected_map.get(base_name, []))
        total_count = len(current_data["stylized_paths"])
        
        return f"📍 {self.current_index + 1}/{len(self.original_keys)} - {current_name} | 已选: {selected_count}/{total_count} 张"
    
    def _get_selected_display(self) -> List[str]:
        """获取所有已选择的图片用于显示"""
        all_selected = []
        for selected_list in self.selected_map.values():
            all_selected.extend(selected_list)
        return all_selected
    
    def _generate_stylized_display(self, stylized_paths: List[str], base_name: str):
        """生成转绘图片的复选框和图片显示"""
        selected_paths = self.selected_map.get(base_name, [])
        
        checkboxes = []
        images = []
        
        for i in range(6):  # 最多显示6张图片
            if i < len(stylized_paths):
                path = stylized_paths[i]
                is_selected = path in selected_paths
                checkboxes.append(gr.update(visible=True, value=is_selected, label=f"版本 {i+1}"))
                images.append(gr.update(visible=True, value=path))
            else:
                checkboxes.append(gr.update(visible=False, value=False))
                images.append(gr.update(visible=False, value=None))
        
        return checkboxes, images
    
    def navigate(self, direction: str) -> Tuple:
        """导航到上一张/下一张"""
        if not self.original_keys:
            return self._get_current_display_data()
        
        if direction == "next":
            self.current_index = (self.current_index + 1) % len(self.original_keys)
        elif direction == "prev":
            self.current_index = (self.current_index - 1) % len(self.original_keys)
        
        return self._get_current_display_data()
    
    def _get_current_display_data(self) -> Tuple:
        """获取当前显示的数据"""
        if not self.original_keys:
            empty_checkboxes = [gr.update(visible=False, value=False) for _ in range(6)]
            empty_images = [gr.update(visible=False, value=None) for _ in range(6)]
            return ("无数据", None, [], *empty_checkboxes, *empty_images)
        
        current_key = self.original_keys[self.current_index]
        current_data = self.indexed_data[current_key]
        self.current_stylized_paths = current_data["stylized_paths"]
        nav_info = self._get_nav_info()
        selected_display = self._get_selected_display()
        
        # 生成复选框和图片显示
        checkboxes, images = self._generate_stylized_display(current_data["stylized_paths"], current_data["base_name"])
        
        return (
            nav_info,
            current_data["original_path"],
            selected_display,
            *checkboxes,
            *images
        )
    
    def update_selection(self, checkbox_states: List[bool]) -> Tuple:
        """根据复选框状态更新选择"""
        if not self.original_keys:
            return "❌ 无数据", [], "无数据"
        
        current_key = self.original_keys[self.current_index]
        current_data = self.indexed_data[current_key]
        base_name = current_data["base_name"]
        
        # 更新选择
        new_selections = []
        for i, is_checked in enumerate(checkbox_states):
            if is_checked and i < len(self.current_stylized_paths):
                new_selections.append(self.current_stylized_paths[i])
        
        self.selected_map[base_name] = new_selections
        
        selected_display = self._get_selected_display()
        nav_info = self._get_nav_info()
        
        if new_selections:
            status = f"✅ 已为 {os.path.basename(current_data['original_path'])} 选择 {len(new_selections)} 张图片"
        else:
            status = f"🔄 已清空 {os.path.basename(current_data['original_path'])} 的选择"
        
        return status, selected_display, nav_info
    
    def remove_selected_image(self, evt: gr.SelectData) -> Tuple:
        """从已选列表中移除图片"""
        if evt.value is None:
            return "❌ 无效的选择", [], "无数据"
        
        # 获取要移除的图片路径
        remove_path = self._extract_image_path(evt.value)
        if not remove_path:
            return "❌ 无法获取图片路径", [], "无数据"
        
        # 从所有选择中移除
        removed = False
        for base_name, selected_list in self.selected_map.items():
            if remove_path in selected_list:
                selected_list.remove(remove_path)
                removed = True
                break
        
        if removed:
            status = f"✅ 已移除: {os.path.basename(remove_path)}"
        else:
            status = f"❌ 未找到要移除的图片: {os.path.basename(remove_path)}"
        
        selected_display = self._get_selected_display()
        nav_info = self._get_nav_info()
        
        # 更新当前页面的复选框状态
        current_key = self.original_keys[self.current_index]
        current_data = self.indexed_data[current_key]
        checkboxes, _ = self._generate_stylized_display(current_data["stylized_paths"], current_data["base_name"])
        
        return status, selected_display, nav_info, *checkboxes
    
    def _extract_image_path(self, value) -> Optional[str]:
        """从选择事件中提取图片路径"""
        if isinstance(value, str):
            return value
        elif isinstance(value, dict):
            if 'image' in value and isinstance(value['image'], dict):
                return value['image'].get('path')
            return value.get('path')
        return None
    
    def clear_all_selections(self) -> Tuple:
        """清空所有选择"""
        total_cleared = sum(len(selected_list) for selected_list in self.selected_map.values())
        self.selected_map = {base_name: [] for base_name in self.selected_map.keys()}
        
        status = f"✅ 已清空所有 {total_cleared} 个选择"
        nav_info = self._get_nav_info()
        
        # 更新当前页面的复选框状态
        if self.original_keys:
            current_key = self.original_keys[self.current_index]
            current_data = self.indexed_data[current_key]
            checkboxes, _ = self._generate_stylized_display(current_data["stylized_paths"], current_data["base_name"])
        else:
            checkboxes = [gr.update(value=False) for _ in range(6)]
        
        return status, [], nav_info, *checkboxes
    
    def save_selected_images(self, output_folder: str) -> str:
        """保存选择的图片"""
        if not output_folder:
            return "❌ 请指定输出文件夹路径"
        
        # 创建输出文件夹
        try:
            os.makedirs(output_folder, exist_ok=True)
        except Exception as e:
            return f"❌ 创建输出文件夹失败: {e}"
        
        # 统计信息
        all_selected = self._get_selected_display()
        if not all_selected:
            return "❌ 没有选择任何图片进行保存"
        
        # 保存图片
        saved_count = 0
        error_messages = []
        
        for stylized_path in all_selected:
            try:
                filename = os.path.basename(stylized_path)
                dest_path = os.path.join(output_folder, filename)
                
                # 避免文件名冲突
                counter = 1
                name, ext = os.path.splitext(dest_path)
                while os.path.exists(dest_path):
                    dest_path = f"{name}_{counter}{ext}"
                    counter += 1
                
                shutil.copy2(stylized_path, dest_path)
                saved_count += 1
                
            except Exception as e:
                error_messages.append(f"保存 {os.path.basename(stylized_path)} 失败: {e}")
        
        # 生成结果消息
        result = f"✅ 保存完成！成功保存 {saved_count}/{len(all_selected)} 张图片到:\n📁 {output_folder}"
        if error_messages:
            result += "\n\n❌ 错误信息:\n" + "\n".join(error_messages)
        
        return result

# 创建全局选择器实例
selector = ImageSelector()

# --- Gradio 界面 ---
def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="🎨 动漫转绘图片挑选工具",
        css="""
        /* iOS风格全局样式 */
        .gradio-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc;
        }
        
        /* 原图显示区域 - iOS风格卡片 */
        .original-image {
            border-radius: 20px;
            border: none;
            overflow: hidden;
            background: #ffffff;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .original-image:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }
        
        .original-image img {
            object-fit: contain !important;
            width: 100% !important;
            height: auto !important;
            max-height: 600px !important;
        }
        
        /* 转绘图片卡片 - iOS风格设计 */
        .stylized-card {
            background: #ffffff;
            border-radius: 16px;
            border: none;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            overflow: hidden;
            position: relative;
            margin: 8px;
        }
        
        .stylized-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        }
        
        .stylized-card.selected {
            border: 3px solid #007AFF;
            box-shadow: 0 8px 30px rgba(0, 122, 255, 0.25);
            transform: translateY(-2px);
        }
        
        .stylized-card img {
            object-fit: cover !important;
            width: 100% !important;
            height: 280px !important;
            border-radius: 12px 12px 0 0;
        }
        
        /* iOS风格选择指示器 */
        .selection-indicator {
            position: absolute;
            top: 12px;
            right: 12px;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border: 2px solid rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        .selection-indicator.selected {
            background: #007AFF;
            border-color: #007AFF;
            color: white;
        }
        
        .selection-indicator:hover {
            transform: scale(1.1);
        }
        
        /* 版本标签 */
        .version-label {
            position: absolute;
            bottom: 12px;
            left: 12px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            backdrop-filter: blur(10px);
        }
        
        /* iOS风格复选框 */
        .ios-checkbox {
            background: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 12px 16px;
            margin: 12px 8px 8px 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease;
        }
        
        .ios-checkbox input[type="checkbox"] {
            width: 20px;
            height: 20px;
            border-radius: 6px;
            border: 2px solid #d1d5db;
            background: #ffffff;
            transition: all 0.2s ease;
        }
        
        .ios-checkbox input[type="checkbox"]:checked {
            background: #007AFF;
            border-color: #007AFF;
        }
        
        .ios-checkbox label {
            font-weight: 500;
            color: #1f2937;
            margin-left: 8px;
        }
        
        /* 已选图片画廊 - iOS风格 */
        .selected-gallery {
            border-radius: 20px;
            border: none;
            background: #ffffff;
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.1);
            padding: 20px;
            min-height: 200px;
        }
        
        /* 状态框 - iOS风格 */
        .status-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 16px;
            padding: 16px 20px;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }
        
        /* iOS风格按钮 */
        .ios-button {
            border-radius: 12px;
            font-weight: 600;
            min-height: 44px;
            border: none;
            transition: all 0.2s ease;
            font-size: 16px;
        }
        
        .ios-button.primary {
            background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(0, 122, 255, 0.3);
        }
        
        .ios-button.primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0, 122, 255, 0.4);
        }
        
        .ios-button.secondary {
            background: #f1f3f4;
            color: #1f2937;
            border: 1px solid #e5e7eb;
        }
        
        .ios-button.secondary:hover {
            background: #e5e7eb;
            transform: translateY(-1px);
        }
        
        /* 标题样式 */
        .section-title {
            font-size: 20px;
            font-weight: 700;
            color: #1f2937;
            margin: 24px 0 16px 0;
            letter-spacing: -0.5px;
        }
        
        .section-subtitle {
            font-size: 14px;
            color: #6b7280;
            margin: -8px 0 20px 0;
            font-weight: 500;
        }
        
        /* 网格布局优化 */
        .stylized-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            padding: 16px;
        }
        
        /* 响应式设计 */
        @media (max-width: 768px) {
            .original-image img {
                max-height: 400px !important;
            }
            .stylized-card img {
                height: 200px !important;
            }
            .stylized-grid {
                grid-template-columns: 1fr;
            }
        }
        """) as demo:
        
        # 标题和说明
        gr.Markdown("# 🎨 动漫转绘图片挑选工具", elem_classes=["section-title"])
        gr.Markdown("""
        **极简工作流程**：选择文件夹 → 加载索引 → 预览对比 → 勾选心仪版本 → 一键保存
        """, elem_classes=["section-subtitle"])
        
        # 文件夹选择区域
        with gr.Row():
            with gr.Column():
                original_folder = gr.Textbox(
                    label="📁 原图文件夹", 
                    placeholder="拖拽或输入原图文件夹路径...",
                    lines=1
                )
            with gr.Column():
                stylized_folder = gr.Textbox(
                    label="🎨 转绘文件夹", 
                    placeholder="拖拽或输入转绘图片文件夹路径...",
                    lines=1
                )
        
        load_btn = gr.Button("🔍 加载图片", variant="primary", size="lg", elem_classes=["ios-button", "primary"])
        status_text = gr.Textbox(
            label="", 
            interactive=False, 
            lines=2,
            elem_classes=["status-box"],
            show_label=False
        )
        
        # 主要内容区域
        with gr.Row():
            # 左侧：原图区域
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 📸 原图", elem_classes=["section-title"])
                original_image = gr.Image(
                    type="filepath",
                    show_label=False,
                    container=True,
                    elem_classes=["original-image"]
                )
                
                nav_info = gr.Textbox(
                    interactive=False,
                    show_label=False,
                    max_lines=1,
                    elem_classes=["ios-checkbox"]
                )
                
                with gr.Row():
                    prev_btn = gr.Button(
                        "◀ 上一张", 
                        interactive=False,
                        elem_classes=["ios-button", "secondary"],
                        scale=1
                    )
                    next_btn = gr.Button(
                        "下一张 ▶", 
                        interactive=False,
                        elem_classes=["ios-button", "secondary"],
                        scale=1
                    )
            
            # 右侧：转绘版本选择
            with gr.Column(scale=2, min_width=700):
                gr.Markdown("### 🎨 转绘版本", elem_classes=["section-title"])
                gr.Markdown("轻触预览，勾选心仪版本", elem_classes=["section-subtitle"])
                
                # 创建6个图片+复选框组合，采用更现代的布局
                stylized_components = []
                checkbox_components = []
                
                with gr.Row():
                    for i in range(3):  # 第一行3个
                        with gr.Column(scale=1):
                            img = gr.Image(
                                type="filepath",
                                show_label=False,
                                container=True,
                                elem_classes=["stylized-card"],
                                visible=False
                            )
                            checkbox = gr.Checkbox(
                                label=f"版本 {i+1}",
                                visible=False,
                                elem_classes=["ios-checkbox"]
                            )
                            stylized_components.append(img)
                            checkbox_components.append(checkbox)
                
                with gr.Row():
                    for i in range(3, 6):  # 第二行3个
                        with gr.Column(scale=1):
                            img = gr.Image(
                                type="filepath",
                                show_label=False,
                                container=True,
                                elem_classes=["stylized-card"],
                                visible=False
                            )
                            checkbox = gr.Checkbox(
                                label=f"版本 {i+1}",
                                visible=False,
                                elem_classes=["ios-checkbox"]
                            )
                            stylized_components.append(img)
                            checkbox_components.append(checkbox)
        
        # 已选图片管理区域
        gr.Markdown("---")
        gr.Markdown("### 🗂️ 已选图片", elem_classes=["section-title"])
        
        with gr.Row():
            gr.Markdown("轻触图片删除，或批量管理", elem_classes=["section-subtitle"])
            with gr.Column(scale=1, min_width=150):
                clear_all_btn = gr.Button(
                    "🗑️ 清空所有",
                    elem_classes=["ios-button", "secondary"]
                )
        
        selected_gallery = gr.Gallery(
            show_label=False,
            columns=6,
            rows=2,
            height=300,
            allow_preview=True,
            preview=True,
            elem_classes=["selected-gallery"]
        )
        
        # 保存区域
        gr.Markdown("---")
        gr.Markdown("### 💾 导出", elem_classes=["section-title"])
        
        with gr.Row():
            output_folder = gr.Textbox(
                label="输出路径",
                placeholder="选择保存文件夹...",
                scale=4
            )
            save_btn = gr.Button(
                "💾 保存精选", 
                variant="primary", 
                scale=1, 
                interactive=False,
                elem_classes=["ios-button", "primary"]
            )
        
        save_status = gr.Textbox(
            interactive=False,
            lines=3,
            elem_classes=["status-box"],
            show_label=False
        )
        
        # 事件绑定
        load_btn.click(
            fn=selector.load_and_index,
            inputs=[original_folder, stylized_folder],
            outputs=[
                status_text, original_image, nav_info, selected_gallery,
                prev_btn, next_btn, save_btn,
                *checkbox_components, *stylized_components
            ]
        )
        
        prev_btn.click(
            fn=lambda: selector.navigate("prev"),
            outputs=[nav_info, original_image, selected_gallery, *checkbox_components, *stylized_components]
        )
        
        next_btn.click(
            fn=lambda: selector.navigate("next"),
            outputs=[nav_info, original_image, selected_gallery, *checkbox_components, *stylized_components]
        )
        
        # 复选框变化事件
        for checkbox in checkbox_components:
            checkbox.change(
                fn=lambda *args: selector.update_selection(list(args)),
                inputs=checkbox_components,
                outputs=[status_text, selected_gallery, nav_info]
            )
        
        selected_gallery.select(
            fn=selector.remove_selected_image,
            outputs=[status_text, selected_gallery, nav_info, *checkbox_components]
        )
        
        clear_all_btn.click(
            fn=selector.clear_all_selections,
            outputs=[status_text, selected_gallery, nav_info, *checkbox_components]
        )
        
        save_btn.click(
            fn=selector.save_selected_images,
            inputs=[output_folder],
            outputs=[save_status]
        )
        
        # 底部信息
        gr.Markdown("""
        ---
        **💡 设计灵感**：简约而不简单，优雅且高效 · 支持 PNG、JPG、JPEG、BMP、GIF、WEBP 格式
        """, elem_classes=["section-subtitle"])
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )