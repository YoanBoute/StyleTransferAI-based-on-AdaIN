from nicegui import ui
from pathlib import Path
import uuid
import json
import asyncio
import websockets
from websockets.asyncio.client import connect
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from utils.file import File
import base64
from copy import deepcopy
from functools import partial

""" 
------------------------------------------------------
|                       TODO                         |
------------------------------------------------------

- Layout changes
    - Control images dimensions to have always same height style blocks and content block
    - Handle Add style button width depending on number of visible blocks
    - Adapt slider value display min width to fit to the max value
    - Center alpha slider and generate button
    - Add titles to the interface components
- Use Requests and Responses objects to communicate
- Auto adjust style weights to have them summed to 1
- Provide a French / English translation
"""

API_URL = "ws://localhost:8000/generate"

class LabeledSlider :
    """Custom component with linked slider and displayed value, along with a label"""
    def __init__(self, label, min_value, max_value, step, default_value):
        with ui.column().classes('w-full gap-1'):
            ui.label(label).classes('text-sm w-full').props('dense')
            with ui.row().classes('w-full gap-5 flex items-center') :
                self.slider = ui.slider(min=min_value, max=max_value, step=step, value=default_value).classes('flex-1')
                self.slider._props['color'] = 'deep-orange'
                self.value_display = ui.number(value=default_value, precision=2, step=step).bind_value(self.slider, 'value').style('width: 70px; min-width: 70px;').props('dense outlined')  

    @property
    def value(self) :
        return self.slider.value
    
    @value.setter
    def value(self, value) :
        self.slider.value = value 

    @property
    def enabled(self) :
        return self.slider.enabled
    
    @enabled.setter
    def enabled(self, value) :
        self.slider.enabled = value


class Checkbox :
    def __init__(self, text : str = '', *, value : bool = False, on_change = None):
        self.box = ui.checkbox(text=text, value=value, on_change=on_change)
        self.box._props['color'] = 'deep-orange'
    
    @property
    def value(self) :
        return self.box.value
    
    @value.setter
    def value(self, value) :
        self.box.value = value
    
    @property
    def enabled(self) :
        return self.box.enabled
    
    @enabled.setter
    def enabled(self, value) :
        self.box.enabled = value


class ImageComponent :
    """Custom image component to store and display an image file chosen by the user (if interactive)"""
    def __init__(self, user_interactive = True) :
        self.interactive = user_interactive
        self.placeholder_img = 'https://placehold.co/600x400?text=Upload+Image' if self.interactive else 'https://placehold.co/600x400?text=Generated+Image'
        self.file = ui.upload(auto_upload=True).props('accept=image/*')
        self.file.visible = False
        if self.interactive :
            with ui.interactive_image(self.placeholder_img).classes('border-solid border-2 rounded border-orange-500 w-full p-1 cursor-pointer') as self.disp_img :
                self.close_btn = ui.button(icon='close').on('click.stop', self.reset_image).classes('absolute top-2 right-2').props('flat fab color=white')
            self.close_btn.visible = False
            self.disp_img.on('click', self.open_file_box)
        else :
            self.disp_img = ui.interactive_image(self.placeholder_img).classes('border-solid border-2 rounded border-orange-500 w-full p-1')
        self.file.on_upload(self.update_img)
        self.disp_img.bind_source_from(self.file, "value")
    
    def update_img(self, e) :
        image_bytes = e.content.read()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = e.type if e.type else 'image/jpeg'
        base64_url = f"data:{mime_type};base64,{encoded_image}"
        self.source = base64_url
        self.close_btn.visible = True
        e.sender.reset() # Clear cache to avoid problems with re-uploading the same file

    def open_file_box(self) :
        js_code = f"""var uploader = getHtmlElement("{self.file.id}").querySelector("input[type=file]").click();"""
        ui.run_javascript(js_code)
    
    def reset_image(self) :
        self.disp_img.set_source(self.placeholder_img)
        self.close_btn.visible = False
    
    def classes(self, add : str = "", *, remove : str = "") :
        self.disp_img.classes(add=add, remove=remove)
        return self

    @property
    def source(self) :
        return self.disp_img.source
    
    @source.setter
    def source(self, value) :
        self.disp_img.source = value
        self.disp_img.force_reload()


class StyleBlock :
    def __init__(self, visible) :
        with ui.card() as self.block :
            self.img = ImageComponent(user_interactive=True)
            self.weight = LabeledSlider('Weight', 0, 1, 0.01, 1)
            self.scale = LabeledSlider('Scale', 0.1, 3, 0.1, 1)
            self.rmv_btn = ui.button('Remove style').classes('w-full')
            self.rmv_btn._props['color'] = 'deep-orange'
        self.block.visible = visible
        self.rmv_btn.visible = False

    def reset(self) :
        self.img.reset_image()
        self.weight.value = 1
        self.scale.value = 1
    
    def classes(self, add : str = "", *, remove : str = "") :
        self.block.classes(add=add, remove=remove)
        return self
    
    @property
    def visible(self) :
        return self.block.visible
    
    @visible.setter
    def visible(self, value) :
        self.block.visible = value


class StyleBlocksRow :
    def __init__(self, max_styles = 5) :
        self.max_styles = max_styles
        with ui.row(align_items='stretch').classes('w-full gap-0') as self.row :
            self.current_width_class = 'w-1/2'
            self.style_blocks = [StyleBlock(visible=(i == 0)).classes(self.current_width_class) for i in range(self.max_styles)]
            self.add_btn = ui.button('➕ Add style', on_click=self.add_block)
            self.add_btn._props['color'] = 'deep-orange'
        self.num_visible = 1
        for i in range(self.max_styles) :
            self.style_blocks[i].rmv_btn.on_click(partial(self.remove_block, i))
        print(self.style_blocks[0].rmv_btn.on_click)

    def __iter__(self) :
        return self.style_blocks.__iter__()
    
    def __getitem__(self, ix) :
        return self.style_blocks[ix]
    
    def copy_block(self, src_ix, target_ix) :
        src_block = self.style_blocks[src_ix]
        target_block = self.style_blocks[target_ix]
        target_block.img.source = src_block.img.source
        target_block.img.close_btn.visible = src_block.img.close_btn.visible
        target_block.weight.value = src_block.weight.value
        target_block.scale.value = src_block.scale.value

    def add_block(self) :
        self.num_visible += 1
        self.style_blocks[self.num_visible - 1].visible = True
        if self.num_visible > 1 :
            for style_block in self.style_blocks :
                style_block.rmv_btn.visible = True
        width_class_to_remove = self.current_width_class
        self.current_width_class = f'w-1/{min(self.num_visible + 1, self.max_styles)}'
        for i in range(self.num_visible) :
            self.style_blocks[i].classes(self.current_width_class, remove=width_class_to_remove) 
        if self.num_visible == self.max_styles : 
            self.add_btn.visible = False
    
    def remove_block(self, block_ix) :
        for i in range(block_ix, min(self.num_visible, self.max_styles - 1)) :
            self.copy_block(i+1, i)
        self.style_blocks[self.num_visible - 1].reset()
        self.style_blocks[self.num_visible - 1].visible = False
        self.num_visible -= 1
        width_class_to_remove = self.current_width_class
        self.current_width_class = f'w-1/{min(self.num_visible + 1, self.max_styles)}'
        for i in range(self.num_visible) :
            self.style_blocks[i].classes(self.current_width_class, remove=width_class_to_remove) 
        if self.num_visible <= 1 : 
            for style_block in self.style_blocks :
                style_block.rmv_btn.visible = False
        self.add_btn.visible = True

    def get_params_dict(self) :
        style_weights = []
        style_scales = []
        for block in self.style_blocks :
            if block.visible and block.img.source is not None :
                style_weights.append(block.weight.value)
                style_scales.append(block.scale.value)
        return {
            "style_weights" : style_weights,
            "style_scales" : style_scales
        }
    

class ParamsBlock :
    def __init__(self):
        with ui.column().classes('w-full') as self.block :
            with ui.column().classes('w-4/5 flex flex-row justify-center') :
                self.alpha = LabeledSlider('Importance of stylization', 0, 1, 0.05, 1)
            with ui.row().classes('w-full flex justify-around') :
                with ui.column(align_items='stretch').classes('w-2/5') :
                    self.preserve_colors = Checkbox('Preserve colors')
                    with ui.card().classes('w-full') :
                        self.resize_size = LabeledSlider('Resize size', 256, 2000, 1, 1000)
                        self.keep_aspect_ratio = Checkbox('Keep aspect ratio')
                with ui.card().classes('w-2/5') :
                    self.patches = Checkbox("Work with patches", value=False, on_change=self.toggle_patches_params)
                    self.patch_size = LabeledSlider("Patch size", 256, 1000, 1, 256)
                    self.patch_context_size = LabeledSlider("Patch context size", 256, 1500, 1, 300)
                    self.patch_overlap = LabeledSlider("Patch overlap", 0, 0.9, 0.05, 0.5)
        
        self.patch_size.enabled = False
        self.patch_context_size.enabled = False
        self.patch_overlap.enabled = False
    
    def toggle_patches_params(self) :
        self.patch_size.enabled = not self.patch_size.enabled
        self.patch_context_size.enabled = not self.patch_context_size.enabled 
        self.patch_overlap.enabled = not self.patch_overlap.enabled
    
    def get_params_dict(self) :
        return {key : component.value for key, component in self.__dict__.items() if key != "block"}
                        

class StyleTransferApp:
    def __init__(self, api_endpoint = API_URL) :
        self.client_id = uuid.uuid4().hex
        self.api_endpoint = api_endpoint

        with ui.row().classes('w-full justify-between') :
            with ui.card().classes('w-7/12') :  
                self.content = ImageComponent()  
                self.style_blocks = StyleBlocksRow()
                self.params_block = ParamsBlock()     
            with ui.column().classes('w-2/5') :
                with ui.card().classes('w-full') :
                    self.generated_img = ImageComponent(False)
                    self.gen_btn = ui.button('Generate image', on_click=self.generate_img).classes('flex justify-center')
                    self.gen_btn._props['color'] = 'deep-orange'
                with ui.card().classes('w-full') :
                    self.status_message = ui.textarea("Status").classes('w-full')
                    self.status_message.props('readonly')
                    self.info_message = ui.textarea("Message").classes('w-full')
                    self.info_message.props('readonly')
    
    def get_params_dict(self) :
        content = File.from_url(self.content.source).model_dump()
        styles = [File.from_url(s.img.source).model_dump() for s in self.style_blocks if s.visible and s.img.source is not None]
        params = self.style_blocks.get_params_dict() | self.params_block.get_params_dict()
        return {
            "client_id" : self.client_id,
            "content" : content,
            "style" : styles,
            "params" : params
        }
    
    async def generate_img(self) :
        request = self.get_params_dict()
        async with connect(self.api_endpoint, max_size=50*1024*1024) as websocket :
            task = await asyncio.create_task(self.handle_request(websocket, request))

    async def handle_request(self, connection, request) :
        try :
            await connection.send(json.dumps(request))
            try:
                response = await connection.recv()
                print("Response received !")
                msg = json.loads(response)
                if msg.get("status") in ("success"):
                    gen_file = File.from_dict(msg["generated_image"])
                    gen_img_path = gen_file.save_to(Path('D:/StyleTransferAI/StyleTransferAI_AdaIN/StyleTransferAI-based-on-AdaIN/style_transfer_api/tmp/resp'))
                    self.generated_img.source = gen_img_path
                self.status_message.value = msg.get("status")
                self.info_message.value = msg.get("message")
            except websockets.ConnectionClosed:
                self.generated_img.source = None
                self.status_message.value = "Connection closed"
                self.info_message.value = None
        except Exception as e :
            print(e)


def main():
    app = StyleTransferApp()
    ui.dark_mode(True)
    ui.run(title="StyleTransferAI", host="0.0.0.0", port=8501)

if __name__ in {"__main__", "__mp_main__"} :
    main()