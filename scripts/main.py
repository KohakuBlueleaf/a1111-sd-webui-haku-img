from modules import scripts
from modules import script_callbacks, shared
from modules import generation_parameters_copypaste as gpc
from modules.ui_components import FormRow

import gradio as gr

from PIL import Image, ImageFilter, ImageEnhance, ImageColor
import cv2
import numpy as np

from hakuimg import blend
from hakuimg import color
from hakuimg import blur
from hakuimg import sketch


'''
UI part
'''

inpaint_base: gr.Image
inpaint_mask: gr.Image
all_btns: list[tuple[gr.Button, ...]] = []
layers = 5


class Script(scripts.Script):
    def title(self):
        return "HakuBlend"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        global img_src, all_btns, inpaint_base, inpaint_mask
        if isinstance(component, gr.Gallery):
            if component.elem_id in {'txt2img_gallery', 'img2img_gallery'}:
                img_src = component
        
        val = kwargs.get("value", "")
        id = kwargs.get("elem_id", "")
        if id=='img_inpaint_base':
            inpaint_base = component
        if id=='img_inpaint_mask':
            inpaint_mask = component
        
        if val == "Send to extras":
            with gr.Accordion('HakuImg', open=False):
                with gr.Column():
                    with gr.Accordion('>> 疊圖', open=False):
                        btns = []
                        for i in range(layers):
                            btn1 = gr.Button(f">> 圖層{i}")
                            btn1.click(None, _js=f"switch_to_haku_img")
                            btns.append(btn1)
                with gr.Column():
                    btn3 = gr.Button(">> 效果")
                    btn3.click(None, _js=f"switch_to_haku_img_eff")
            
            all_btns.append((btns, btn3, img_src))
    
    def ui(self, is_img2img):
        return []


def add_tab():
    print('add tab')
    with gr.Blocks(analytics_enabled=False) as demo:
        with FormRow().style(equal_height=False):
            with gr.Column():
                with gr.Tabs(elem_id="haku_blend_tabs"):
                    with gr.TabItem('Blend', elem_id='haku_blend'):
                        all_layers = []
                        all_alphas = []
                        all_mask_str = []
                        all_mask_blur = []
                        img_blend_h_slider = gr.Slider(160, 1280, 320, step=10, label="Image preview height", elem_id='haku_img_h_blend')
                        with gr.Tabs(elem_id="haku_blend_layers_tabs"):
                            for i in range(1, layers+1):
                                with gr.TabItem(f'Layer{i}', elem_id=f'haku_blend_Layer{i}'):
                                    all_layers.append(
                                        gr.ImageMask(type='numpy', label=f"Layer{i}", elem_id=f'haku_img_blend{i}')
                                    )
                                    all_alphas.append(
                                        gr.Slider(0, 1, 0.5, label=f"Layer{i} transparent")
                                    )
                                    all_mask_blur.append(
                                        gr.Slider(0, 32, 4, label=f"Layer{i} mask blur")
                                    )
                                    all_mask_str.append(
                                        gr.Slider(0, 1, 1, label=f"Layer{i} mask strength")
                                    )
                        bg_color = gr.ColorPicker('#FFFFFF', label='background color')
                        expand_btn = gr.Button("refresh", variant="primary")
                    
                    with gr.TabItem('Effect', elem_id='haku_eff'):
                        img_eff_h_slider = gr.Slider(160, 1280, 320, step=10, label="Image preview height", elem_id='haku_img_h_eff')
                        image_eff = gr.Image(type='numpy', label="img", elem_id='haku_img_eff', show_label=False)
                        with gr.Tabs(elem_id='effect_tabs'):
                            with gr.TabItem('Color', elem_id='haku_color'):
                                temp_slider = gr.Slider(-100, 100, 0, step=1, label="temparature")
                                hue_slider = gr.Slider(-90, 90, 0, step=1, label="hue")
                                bright_slider = gr.Slider(-100, 100, 0, step=1, label="brightness")
                                contrast_slider = gr.Slider(-100, 100, 0, step=1, label="contrast")
                                sat_slider = gr.Slider(-100, 100, 0, step=1, label="saturation")
                                gamma_slider = gr.Slider(0.2, 2.2, 1, step=0.1, label="Gamma")
                                color_btn = gr.Button("refresh", variant="primary")
                            
                            with gr.TabItem('Blur', elem_id='haku_blur'):
                                blur_slider = gr.Slider(0, 32, 8, label="blur")
                                blur_btn = gr.Button("refresh", variant="primary")
                            
                            with gr.TabItem('Sketch', elem_id='haku_sketch'):
                                sk_sigma = gr.Slider(1, 5, 1.4, step=0.05, label='sigma')
                                sk_k_sigma = gr.Slider(1, 5, 1.6, step=0.05, label='k_sigma')
                                sk_eps = gr.Slider(-0.2, 0.2, -0.03, step=0.005, label='epsilon')
                                sk_phi = gr.Slider(1, 50, 10, step=1, label='phi')
                                sk_gamma = gr.Slider(0.75, 1, 1, step=0.005, label='gamma')
                                sk_color = gr.Radio(['gray', 'rgb'], value='gray', label='color mode')
                                sk_scale = gr.Checkbox(False, label='use scale')
                                sketch_btn = gr.Button("refresh", variant="primary")
                    
                    with gr.TabItem('Other'):
                        with gr.Tabs(elem_id='function list'):
                            with gr.TabItem('InOutPaint'):
                                iop_mask = gr.Image(type='pil', visible=False)
            
            with gr.Column():
                image_out = gr.Image(
                    interactive=False, 
                    type='pil', 
                    label="haku_output", 
                    elem_id='haku_out'
                )
                with gr.Row():
                    send_btns = gpc.create_buttons(["img2img", "inpaint", "extras"])
                    send_ip_b = gr.Button("Send to inpaint upload", elem_id='send_inpaint_base')
                with gr.Row():
                    with gr.Accordion('Send to Blend', open=False):
                        send_blends = []
                        for i in range(1, layers+1):
                            send_blends.append(gr.Button(f"Send to Layer{i}", elem_id=f'send_haku_blend{i}'))
                    send_eff = gr.Button("Send to Effect", elem_id='send_haku_blur')
        img_blend_h_slider.change(None, img_blend_h_slider, _js=f'get_change_height("haku_img_blend")')
        img_eff_h_slider.change(None, img_eff_h_slider, _js=f'get_change_height("haku_img_eff")')
        
        # blend
        all_blend_set = [bg_color]
        all_blend_set += all_alphas+all_mask_blur+all_mask_str
        all_blend_input = all_blend_set + all_layers
        for component in all_blend_set:
            component.change(blend.run(layers), all_blend_input, image_out)
        expand_btn.click(blend.run(layers), all_blend_input, image_out)
        
        #blur
        all_blur_input = [image_eff, blur_slider]
        blur_slider.change(blur.run, all_blur_input, outputs=image_out)
        blur_btn.click(blur.run, all_blur_input, outputs=image_out)
        
        #color
        all_color_set = [
            bright_slider, contrast_slider, sat_slider, 
            temp_slider, hue_slider, gamma_slider
            
        ]
        all_color_input = [image_eff] + all_color_set
        for component in all_color_set:
            component.change(color.run, all_color_input, image_out)
        color_btn.click(color.run, all_color_input, image_out)
        
        #sketch
        all_sk_set = [
            sk_sigma, sk_k_sigma, sk_eps, sk_phi, sk_gamma, sk_color, sk_scale
        ]
        all_sk_input = [image_eff] + all_sk_set
        for component in all_sk_set:
            component.change(sketch.run, all_sk_input, image_out)
        sketch_btn.click(sketch.run, all_sk_input, image_out)
        
        #send
        for btns, btn3, img_src in all_btns:
            for btn, img in zip(btns, all_layers):
                btn.click(gpc.image_from_url_text, img_src, img, _js="extract_image_from_gallery")
            btn3.click(gpc.image_from_url_text, img_src, image_eff, _js="extract_image_from_gallery")
        
        gpc.bind_buttons(send_btns, image_out, None)
        for btn, img in zip(btns, all_layers):
            btn.click(lambda x:x, image_out, img)
            btn.click(None, _js = 'switch_to_haku_blend')
        
        for layer, send_btn in zip(all_layers, send_blends):
            send_btn.click(lambda x:x, image_out, layer)
            send_btn.click(None, _js='switch_to_haku_blend')
        
        send_ip_b.click(lambda *x:x, [image_out, iop_mask], [inpaint_base, inpaint_mask])
        send_ip_b.click(None, _js = 'switch_to_inpaint_upload')
            
        send_eff.click(lambda x:x, image_out, image_eff)
        send_eff.click(None, _js = 'switch_to_haku_eff')
        
    return (demo , "HakuImg", "haku_img"),


script_callbacks.on_ui_tabs(add_tab)