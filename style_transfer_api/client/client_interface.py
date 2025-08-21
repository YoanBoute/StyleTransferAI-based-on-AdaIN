from nicegui import ui
from pathlib import Path
import uuid
import json
import asyncio
import websockets
from websockets.asyncio.client import connect
from utils.file import File
from utils.requests import Request, GenRequest, CancelRequest, Response, GenerationStatus
import base64
from functools import partial
from datetime import datetime

""" 
------------------------------------------------------
|                       TODO                         |
------------------------------------------------------

- Layout changes
    - Find a better way to print generation status (and eventually message) -> As a notification maybe ?
    - Color the background of the warning card and change its font
    - Change placeholders for images
- Connect to the server on interface launch, and delete all client-related files on the server when this connection is closed
- Bug : When removing the first style block while it has an image and the second has no image, the new first will keep the "change image" button and images won't display anymore
- Bug : Sometimes, an image won't be uploaded, which breaks the image component (randomly)
- Add all necessary warnings (including a check to see whether the server is connected)
- Add a "Reset weights" button
- Handle invalid requests (missing style / content)
- Provide a French / English translation
- Add tooltips with information logo next to each parameter
- Add a global description message dialog that indicates how to use the interface
"""

API_URL = "ws://localhost:8000/generate"
TMP_FILES_PATH = Path('D:/StyleTransferAI/StyleTransferAI_AdaIN/StyleTransferAI-based-on-AdaIN/style_transfer_api/tmp/')

ui.add_head_html('''
    <style>
    .q-dialog__inner {
        padding: 0 !important;
    }
    .fullscreen-dialog .q-card {
        max-width: none !important;
        max-height: none !important;
        width: 100% !important;
        height: 100% !important;
        padding: 5vh 5vw !important;
        background: none !important;
        box-shadow: none !important;
        backdrop-filter: blur(5px) !important;
    }
    .fullscreen-dialog > .q-dialog__backdrop {
        background: rgba(0,0,0,0.8) !important;
    }
    .img-contain img {
        object-fit: contain !important;
    }
    </style>
    ''')

class LabeledSlider :
    """Custom component with linked slider and displayed value, along with a label"""
    def __init__(self, label, min_value, max_value, step, default_value, fix_option = False):
        with ui.column().classes('w-full gap-1') as self.block :
            ui.label(label).classes('text-sm w-full').props('dense')
            with ui.row().classes('w-full gap-5 flex items-center') :
                self.slider = ui.slider(min=min_value, max=max_value, step=step, value=default_value).classes('flex-1')
                self.slider._props['color'] = 'deep-orange'
                self.value_display = ui.number(value=default_value, precision=2, step=step, format="%i" if type(step) is int else f"%.{len(str(step))-2}f").bind_value(self.slider, 'value').style(f'width:{40+8*max(len(str(step)), len(str(max_value)))}px').props('dense outlined')  
                if fix_option :
                    self.fix_btn = TogglableButton(icon1='lock_open', icon2='lock')

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
        self.value_display.enabled = value
    
    @property 
    def visible(self) :
        return self.block.visible
    
    @visible.setter
    def visible(self, value) :
        self.block.visible = value
    
    @property
    def is_fixed(self) :
        if self.__dict__.get("fix_btn") is None :
            return False
        return self.fix_btn.value


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
    
    def on_value_change(self, handler) :
        self.box.on_value_change(handler)


class TogglableButton :
    def __init__(self, icon1, icon2):
        self.btn = ui.button(icon=icon1, on_click=self.toggle).props('round color=deep-orange')
        self._value = False
        self._icon1 = icon1
        self._icon2 = icon2
    
    def toggle(self) :
        self.value = not self.value
    
    @property
    def value(self) :
        return self._value
    
    @value.setter
    def value(self, new_value) :
        self._value = new_value
        self.btn.icon = self._icon2 if self.value else self._icon1
        self.btn.props(f'color={"deep-orange-10" if self.value else "deep-orange"}')


class ImageComponent :
    """Custom image component to store and display an image file chosen by the user (if interactive)"""
    def __init__(self, user_interactive = True, source = None) :
        self.interactive = user_interactive
        self.placeholder_img = 'https://placehold.co/600x400?text=Upload+Image' if self.interactive else 'https://placehold.co/600x400?text=Generated+Image'
        self.file = ui.upload(auto_upload=True).props('accept=image/*')
        self.file.visible = False
        if self.interactive :
            with ui.interactive_image(self.placeholder_img).classes('border-solid border-2 rounded border-orange-500 p-1 cursor-pointer img-contain') as self.disp_img :
                self.close_btn = ui.button(icon='close').on('click.stop', self.reset_image).classes('absolute top-2 right-2').props('round flat color=grey-5')
            self.close_btn.visible = False
            self.disp_img.on('click.stop', lambda : self.open_file_box() if not self.valid_img else None)
        else :
            with ui.interactive_image(self.placeholder_img).classes('border-solid border-2 rounded border-orange-500 p-1 img-contain') as self.disp_img :
                self.dwnld_btn = ui.button(icon='download').on('click.stop', self.download).classes('absolute top-2 right-2').props('round flat color=grey-5')
            self.dwnld_btn.visible = False
        self.disp_img.on('click.stop', lambda : self.open_fullscreen() if self.valid_img else None)
        self.file.on_upload(self.update_img)

        self.dialog = ui.dialog().classes('fullscreen-dialog')
        with self.disp_img :
            self.change_img_btn = ui.button(icon='photo_library').on('click.stop', self.open_file_box).classes('absolute top-2 right-10').props('round flat color=grey-5')
        self.change_img_btn.visible = False
        if source is not None :
            self.source = source

    def update_img(self, e) :
        image_bytes = e.content.read()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = e.type if e.type else 'image/jpeg'
        base64_url = f"data:{mime_type};base64,{encoded_image}"
        self.source = base64_url
        self.close_btn.visible = True
        self.change_img_btn.visible = True
        e.sender.reset() # Clear cache to avoid problems with re-uploading the same file
        ui.run_javascript(f"emitEvent('image-{self.disp_img.id}-update')")
    
    def open_fullscreen(self) :
        self.dialog.clear()
        with self.dialog:
            with ui.card().classes('bg-black p-0 cursor-pointer').on('click.stop', self.dialog.close) :
                self.fs_close_btn = ui.button(icon='close').classes('absolute top-2 right-2 z-50').props('round flat color=grey-5')
                if not self.interactive :
                    self.fs_dwnld_btn = ui.button(icon='download').on('click.stop', self.download).classes('absolute top-2 right-12 z-50').props('round flat color=grey-5')
                self.fs_img = ui.image(self.source).classes('img-contain')
            self.dialog.open()

    def open_file_box(self) :
        js_code = f"""var uploader = getHtmlElement("{self.file.id}").querySelector("input[type=file]").click();"""
        ui.run_javascript(js_code)
    
    def reset_image(self) :
        self.disp_img.set_source(self.placeholder_img)
        self.close_btn.visible = False
        self.change_img_btn.visible = False
        ui.run_javascript(f"emitEvent('image-{self.disp_img.id}-reset')")

    def download(self) :
        ui.download(self.source, filename=f"StyleTransfer_{datetime.today().strftime("%Y%m%d_%H-%M-%S")}")
    
    def classes(self, add : str = "", *, remove : str = "") :
        self.disp_img.classes(add=add, remove=remove)
        return self

    @property
    def source(self) :
        return self.disp_img.source
    
    @source.setter
    def source(self, value) :
        self.disp_img.source = value
        if not self.interactive :
            if value != self.placeholder_img :
                self.classes('cursor-pointer')
                self.dwnld_btn.visible = True
            else :
                self.classes(remove='cursor-pointer')
                self.dwnld_btn.visible = False
        self.disp_img.force_reload()
    
    @property
    def valid_img(self) :
        return self.disp_img.source != self.placeholder_img
    

class ImageCarousel :
    def __init__(self, num_img_per_page = 3) :
        # with ui.carousel(animated=True, arrows=True) as self.carousel :
        #     # for i in range(0, 9, 3) :
        #         with ui.carousel_slide() :
        #             with ui.row(wrap=False) :
        #                 ui.image(f'https://picsum.photos/id/4/270/180').classes('w-96')
        #                 ui.image(f'https://picsum.photos/id/44/270/180').classes('w-96')
        #                 ui.image(f'https://picsum.photos/id/444/270/180').classes('w-96')
        with ui.card() as self.block :
            ui.label('Generation history').classes('font-medium text-slate-200 h-[5%] m-0 p-0')
            self.carousel = ui.carousel(animated=True, arrows=True).classes('h-[94%] w-[99%] m-auto')
        self.sources = []
        self.num_img_per_page = num_img_per_page
        self.block.bind_visibility_from(self, 'has_images')

    @property
    def has_images(self) :
        return len(self.sources) > 0
    
    @property
    def visible(self) :
        return self.block.visible
    
    def classes(self, add : str = "", *, remove : str = "") :
        self.block.classes(add=add, remove=remove)
        return self

    def add_image(self, source) :
        self.sources.insert(0, source)
        self.update()
    
    def update(self) :
        self.carousel.clear()
        num_images = len(self.sources)
        num_pages = num_images // self.num_img_per_page if num_images % self.num_img_per_page == 0 else (num_images // self.num_img_per_page) + 1
        with self.carousel :
            for i in range(num_pages) :
                with ui.carousel_slide() :
                    with ui.row().classes('w-full h-full gap-1 justify-center overflow-hidden') :
                        for j in range(i*self.num_img_per_page, min(num_images, (i+1) * self.num_img_per_page)) :
                            ImageComponent(user_interactive=False, source=self.sources[j]).classes(f'w-[{(100/self.num_img_per_page) - 1}%] h-full')


class StyleBlock :
    def __init__(self, visible) :
        with ui.card() as self.block :
            self.img = ImageComponent(user_interactive=True).classes("h-[25vh] w-full")
            self.weight = LabeledSlider('Weight', 0, 1, 0.01, 1, fix_option=True)
            self.weight.visible = False # Weight slider should only appear if multiple styles are displayed
            self.weight.enabled = False
            self.scale = LabeledSlider('Scale', 0.1, 3, 0.1, 1)
            self.scale.enabled = False
            self.rmv_btn = ui.button('Remove style').classes('w-full')
            self.rmv_btn._props['color'] = 'deep-orange'
            self.rmv_btn.visible = False
        self.block.visible = visible
        ui.on(f"image-{self.img.disp_img.id}-update", self.activate_sliders)
        ui.on(f"image-{self.img.disp_img.id}-reset", self.deactivate_sliders)

    def reset(self) :
        self.img.reset_image()
        self.weight.value = 1
        self.weight.fix_btn.value = False
        self.scale.value = 1
    
    def activate_sliders(self) :
        self.weight.enabled = True
        self.scale.enabled = True
        ui.run_javascript(f"emitEvent('weights-{self.weight.slider.id}-added')")

    def deactivate_sliders(self) :
        self.weight.enabled = False
        self.scale.enabled = False
        ui.run_javascript("emitEvent('weights-removed')")
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

    @property
    def is_valid(self) :
        return self.visible and self.img.valid_img


class StyleBlocksRow :
    def __init__(self, max_styles = 5) :
        self.max_styles = max_styles
        ui.label('Style images').classes('font-medium text-slate-200')
        with ui.row(align_items='center').classes('w-full justify-around gap-0') as self.row :
            self.current_width_class = 'w-1/2'
            self.style_blocks = [StyleBlock(visible=(i == 0)).classes(self.current_width_class) for i in range(self.max_styles)]
            self.add_btn = ui.button('Add style', icon='add_circle', on_click=self.add_block)
            self.add_btn._props['color'] = 'deep-orange'
        self.num_visible = 1
        for i in range(self.max_styles) :
            self.style_blocks[i].rmv_btn.on_click(partial(self.remove_block, i))
            self.style_blocks[i].weight.slider.on("change", partial(self.update_weight_values, i, True))
            self.style_blocks[i].weight.value_display.on("change", partial(self.update_weight_values, i, True))
            ui.on(f"weights-{self.style_blocks[i].weight.slider.id}-added", partial(self.update_weight_values, i, block_added = True))
        ui.on("weights-removed", self.update_weight_values)

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
                style_block.weight.visible = True
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
                style_block.weight.visible = False
        self.add_btn.visible = True
        self.update_weight_values()

    def update_weight_values(self, style_ix = None, fix_value = False, block_added = False) :
        """When the value of a weight slider is changed, proportionnally adjust the others to maintain a sum of 1

        Args:
            style_ix (int): Index of the updated style weight
            new_weight (float): New value of the updated style weight
        """
        if fix_value :
            self.style_blocks[style_ix].weight.fix_btn.value = True

        previous_values = {}
        fixed_weights = []
        for i in range(self.num_visible) :
            if i == style_ix or not self.style_blocks[i].is_valid :
                continue
            if self.style_blocks[i].weight.is_fixed :
                fixed_weights.append(self.style_blocks[i].weight.value)
            else :
                previous_values[i] = self.style_blocks[i].weight.value
        
        total_fixed_weight = sum(fixed_weights)
        # if style_ix is not None :
        #     self.style_blocks[style_ix].weight.value = min(self.style_blocks[style_ix].weight.value, 1 - total_fixed_weight)
        remaining_weight = 1 - self.style_blocks[style_ix].weight.value if style_ix is not None and not block_added else 1
        remaining_weight -= total_fixed_weight
        remaining_weight = max(0, remaining_weight)
        
        previous_total_weight = sum(list(previous_values.values()))
        if block_added and len(set(previous_values.values())) == 1 :
            new_value = remaining_weight / (len(previous_values) + 1)
            self.style_blocks[style_ix].weight.value = new_value
            for i in previous_values.keys() :
                self.style_blocks[i].weight.value = new_value
        elif previous_total_weight == 0 :
            for i, prev_value in previous_values.items() :
                self.style_blocks[i].weight.value = remaining_weight / len(previous_values)
        else :
            for i, prev_value in previous_values.items() :
                self.style_blocks[i].weight.value = prev_value * remaining_weight / previous_total_weight

    def get_params_dict(self) :
        style_weights = []
        style_scales = []
        for block in self.style_blocks :
            if block.is_valid :
                style_weights.append(block.weight.value)
                style_scales.append(block.scale.value)
        return {
            "style_weights" : style_weights,
            "style_scales" : style_scales
        }
    

class ParamsMenu :
    def __init__(self):
        with ui.right_drawer(value=False, elevated=True, bordered=True).props('overlay').classes('z-50') as self.menu :
            ui.label('Generation parameters').classes('text-xl font-semibold mb-[5vh] mt-2')
            self.close_btn = ui.button(icon='close', on_click=self.toggle).props('round flat color=grey-5').classes('absolute top-4 right-4 z-50')
            with ui.column().classes('w-full gap-5') :
                self.alpha = LabeledSlider('Importance of stylization', 0, 1, 0.05, 1)
                self.preserve_colors = Checkbox('Preserve colors')
                with ui.card().classes('w-full') :
                    self.resize_size = LabeledSlider('Resize size', 256, 2000, 1, 1000)
                    self.keep_aspect_ratio = Checkbox('Keep aspect ratio')
                with ui.card().classes('w-full') :
                    self.patches = Checkbox("Work with patches", value=False, on_change=self.toggle_patches_params)
                    self.patch_size = LabeledSlider("Patch size", 256, 1000, 1, 256)
                    self.patch_context_size = LabeledSlider("Patch context size", 256, 1500, 1, 300)
                    self.patch_overlap = LabeledSlider("Patch overlap", 0, 0.9, 0.05, 0.5)
        self.blur_overlay = ui.element('div').classes('fixed inset-0 backdrop-blur-sm z-40 bg-black/30 hidden').on('click.stop', self.toggle)
        self.patch_size.enabled = False
        self.patch_context_size.enabled = False
        self.patch_overlap.enabled = False
    
    def toggle_patches_params(self) :
        self.patch_size.enabled = not self.patch_size.enabled
        self.patch_context_size.enabled = not self.patch_context_size.enabled 
        self.patch_overlap.enabled = not self.patch_overlap.enabled
    
    def toggle(self) :
        self.menu.toggle()
        if self.menu.value :
            self.blur_overlay.classes(remove='hidden')
        else :
            self.blur_overlay.classes(add='hidden')

    def get_params_dict(self) :
        params_dict = {key : component.value for key, component in self.__dict__.items() if key not in ["menu", "close_btn", "blur_overlay"]}
        params_dict['resize_size'] = int(params_dict['resize_size'])
        params_dict['patch_size'] = int(params_dict['patch_size'])
        params_dict['patch_context_size'] = int(params_dict['patch_context_size'])
        return params_dict
                        

class StyleTransferApp:
    def __init__(self, api_endpoint = API_URL) :
        self.client_id = uuid.uuid4().hex
        self.api_endpoint = api_endpoint
        self.menu = ParamsMenu()
        self.menu_btn = ui.button(icon='menu', on_click=self.menu.toggle).props('round color=deep-orange').classes('fixed top-4 right-4 z-50 bg-black/50 hover:bg-black/70 text-white')
        with ui.element('div').classes('w-full h-[10vh] justify-center align-center') :
            ui.label('Style Transfer AI').classes('text-[6vh] w-[30vw] m-auto text-center font-bold font-["Arial Black", Gadget, sans-serif]')
        with ui.row().classes('w-full justify-between') :
            with ui.card().classes('w-[49%]') : 
                ui.label('Content image').classes('font-medium text-slate-200')
                self.content = ImageComponent().classes('h-[50vh] w-full')  
                self.style_blocks = StyleBlocksRow()
            with ui.column().classes('w-[49%]') : 
                with ui.card().classes('w-full') :
                    ui.label('Generated image').classes('font-medium text-slate-200')
                    self.generated_img = ImageComponent(user_interactive=False).classes('h-[50vh] w-full')
                    with self.generated_img.disp_img :
                        self.loading = ui.spinner(type='bars', size='15%', color='deep-orange').classes('absolute inset-0 m-auto z-10')
                        self.blur = ui.element('div').classes('absolute inset-0 backdrop-blur-sm') # Blurs current image while a generation is in progress
                    with ui.row().classes('w-full justify-around') :
                        self.gen_btn = ui.button('Generate image', on_click=self.generate_img).classes('block m-auto')
                        self.gen_btn._props['color'] = 'deep-orange'
                        self.cancel_btn = ui.button('Cancel generation', on_click=self.cancel_gen).classes('block m-auto')
                        self.cancel_btn._props['color'] = 'deep-orange'
                        self.cancel_btn.visible = False
                self.carousel_block = ImageCarousel().classes('w-full h-[30vh]')
                with ui.card().classes('w-full') as self.warning_block :
                    self.warning = ui.textarea("Warning").classes('w-full')
                    self.warning.props("readonly")
                    ui.timer(0.5, self.check_for_warnings)
                with ui.card().classes('w-full') :
                    self.status_message = ui.textarea("Status").classes('w-full')
                    self.status_message.props('readonly')
                    self.info_message = ui.textarea("Message").classes('w-full')
                    self.info_message.props('readonly')
        self.loading.bind_visibility_from(self.cancel_btn)
        self.blur.bind_visibility_from(self.cancel_btn)

    def generate_request(self) :
        content = File.from_url(self.content.source).model_dump()
        styles = [File.from_url(s.img.source).model_dump() for s in self.style_blocks if s.is_valid]
        params = self.style_blocks.get_params_dict() | self.menu.get_params_dict()
        return GenRequest(client_id=self.client_id, content_img=content, style_imgs=styles, params=params)
    
    async def generate_img(self) :
        request = self.generate_request()
        self.cancel_btn.visible = True
        async with connect(self.api_endpoint, max_size=50*1024*1024) as websocket :
            await asyncio.create_task(self.handle_request(websocket, request))

    async def cancel_gen(self) :
        request = CancelRequest(client_id=self.client_id)
        self.cancel_btn.visible = False
        async with connect(self.api_endpoint, max_size=50*1024*1024) as websocket :
            await asyncio.create_task(self.handle_request(websocket, request))

    async def handle_request(self, connection, request : Request) :
        try :
            await connection.send(json.dumps(request.model_dump()))
            try:
                resp = await connection.recv()
                resp_dict = json.loads(resp)
                response = Response.from_dict(resp_dict)
                if response.status in (GenerationStatus.success, GenerationStatus.error) :
                    self.cancel_btn.visible = False
                if response.status == GenerationStatus.success :
                    gen_file = response.generated_image
                    gen_img_path = gen_file.save_to(TMP_FILES_PATH / datetime.today().strftime('%Y%m%d_%H-%M-%S'))
                    self.generated_img.source = gen_img_path
                    self.carousel_block.add_image(gen_img_path)
                self.status_message.value = response.status
                self.info_message.value = response.message
            except websockets.ConnectionClosed:
                self.generated_img.source = None
                self.status_message.value = "Connection closed"
                self.info_message.value = None
        except Exception as e :
            print(e)
    
    def check_for_warnings(self) :
        warnings = ''
        sum_one = self.sum_one_warning()
        if sum_one is not None :
            warnings += f'- {self.sum_one_warning()}\n'

        self.warning.value = warnings
        if warnings == '' :
            self.warning_block.visible = False
        else :
            self.warning_block.visible = True

    def sum_one_warning(self) :
        valid_weights = [s.weight.value for s in self.style_blocks if s.is_valid]
        if len(valid_weights) == 0 or sum(valid_weights) == 1 :
            return None
        else :
            return 'The style weights do not sum to 1'


def main():
    app = StyleTransferApp()
    ui.dark_mode(True)
    ui.run(title="StyleTransferAI", host="0.0.0.0", port=8502, reload=True)

if __name__ in {"__main__", "__mp_main__"} :
    main()