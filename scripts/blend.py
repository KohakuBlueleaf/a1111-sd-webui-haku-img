import gradio as gr
from modules import scripts
from modules import script_callbacks, shared
from modules import generation_parameters_copypaste as gpc
from modules.ui_components import FormRow

from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np


class Script(scripts.Script):
    def title(self):
        return "HakuBlend"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        global image_main, image_mask, img_src, all_btns
        # Add button to both txt2img and img2img tabs
        # print(type(component), kwargs.get("value"), kwargs.keys())
        if isinstance(component, gr.Gallery):
            if component.elem_id in {'txt2img_gallery', 'img2img_gallery'}:
                img_src = component
            print(kwargs.get('label'), kwargs.get('elem_id'), component.elem_id)
        
        # if isinstance(component, gr.Image):
        #     if component.elem_id in {'txt2img_preview', 'img2img_preview'}:
        #         img_src = component
        
        if kwargs.get("value") == "Send to extras":
            with gr.Row(variant='panel'):
                with gr.Column():
                    btn1 = gr.Button(">> 疊圖1")
                    btn1.click(None, _js=f"switch_to_haku_img")
                    
                    btn2 = gr.Button(">> 疊圖2")
                    btn2.click(None, _js=f"switch_to_haku_img")
                with gr.Column():
                    btn3 = gr.Button(">> 模糊")
                    btn3.click(None, _js=f"switch_to_haku_img_blur")
                    
                    btn4 = gr.Button(">> 色彩")
                    btn4.click(None, _js=f"switch_to_haku_img_color")
            
            all_btns.append((btn1, btn2, btn3, btn4, img_src))

    def ui(self, is_img2img):
        return []


all_btns = []
img_src = None
img1: Image.Image = None
img2: Image.Image = None
alpha: float = 0.5
m2_str = 1.0
m2_blur = 4

img3: Image.Image = None
img_blur = 4


global_data = {
    'img1': None,
    'img2': None,
    'img3': None,
    'alpha': 0.5,
    'm2_str': 1.0,
    'm2_blur': 4,
    'img_blur': 8,
    'imgc': None,
    'bright': 0,
    'contrast': 0,
    'sat': 0,
    'hue': 0,
    'temp': 0,
    'gamma': 1
}


def on_change(name, runner=None):
    def foo(x=None, *args):
        global global_data
        if x is not None:
            global_data[name] = x
        if runner is not None:
            return runner(*args)
    return foo


def run(img1, img2, alpha, m2_str, m2_blur):
    print(alpha, m2_str, m2_blur)
    image1 = Image.fromarray(img1)
    # mask1 = Image.fromarray(255-img1['mask'][:,:,0], mode='L')
    image2 = Image.fromarray(img2['image'])
    mask2 = img2['mask'][:,:,0]*m2_str
    mask2 = Image.fromarray((255-mask2).astype(np.uint8), mode='L')
    
    image2 = image2.resize(image1.size)
    mask2 = mask2.resize(image1.size).filter(ImageFilter.GaussianBlur(m2_blur))
    
    image1 = np.array(image1).astype(np.float32)
    # mask1 = np.expand_dims(np.array(mask1)/255, 2)
    image2 = np.array(image2).astype(np.float32)
    mask2 = np.expand_dims(np.array(mask2)/255, 2)
    
    image2 = image1*(1-mask2) + image2*mask2
    res = image1*(1-alpha)+image2*alpha
    res = res.astype(np.uint8)
    return Image.fromarray(res, mode='RGB')


def blur(img3, img_blur):
    img = Image.fromarray(img3)
    blur = ImageFilter.GaussianBlur(img_blur)
    return img.filter(blur)


def run_color(img1, bright, contrast, sat, temp, hue, gamma):
    bright /=100
    contrast /=100
    temp /=100
    sat /=100
    
    #brigtness
    res = Image.fromarray(img1)
    brightness = ImageEnhance.Brightness(res)
    res = brightness.enhance(1+bright)
    
    #contrast
    cont = ImageEnhance.Contrast(res)
    res = cont.enhance(1+contrast)
    res = np.array(res).astype(np.float32)
    
    #temp
    if temp>0:
        res[:, :, 0] *= 1+temp
        res[:, :, 1] *= 1+temp*0.4
    elif temp<0:
        res[:, :, 2] *= 1-temp
    res = np.clip(res, 0, 255)/255
    res = np.clip(np.power(res, gamma), 0, 1)
    
    #saturation
    print(res.shape)
    sat_real = 1 + sat
    hls_img = cv2.cvtColor(res, cv2.COLOR_RGB2HLS)
    hls_img[:, :, 2] = np.clip(sat_real*hls_img[:, :, 2], 0, 1)
    res = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB)*255
    
    # hue
    hsv_img = cv2.cvtColor(res, cv2.COLOR_RGB2HSV)
    print(np.max(hsv_img[:, :, 0]), np.max(hsv_img[:, :, 1]), np.max(hsv_img[:, :, 2]))
    hsv_img[:, :, 0] = (hsv_img[:, :, 0]+hue)%360
    
    res = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    
    res = res.astype(np.uint8)
    res = Image.fromarray(res, mode='RGB')
    return res


def add_tab():
    print('add tab')
    with gr.Blocks(analytics_enabled=False) as demo:
        with FormRow().style(equal_height=False):
            with gr.Column():
                with gr.Tabs(elem_id="haku_blend_tabs"):
                    with gr.TabItem('疊圖', elem_id='haku_blend'):
                        img1_h_slider = gr.Slider(160, 1280, 320, step=10, label="圖1預覽高度", elem_id='haku_img1_h')
                        img2_h_slider = gr.Slider(160, 1280, 320, step=10, label="圖2預覽高度", elem_id='haku_img2_h')
                        image_main = gr.Image(type='numpy', label="圖1", elem_id='haku_img1', show_label=False)
                        image_mask = gr.ImageMask(type='numpy', label="圖2", elem_id='haku_img2', show_label=False)
                        alpha_slider = gr.Slider(0, 1, 0.5, label="圖1透明度")
                        mask2_blur_slider = gr.Slider(0, 32, 4, label="圖2蒙版模糊")
                        mask2_slider = gr.Slider(0, 1, 1, label="圖2蒙版強度")
                        expand_btn = gr.Button("refresh", variant="primary")
                        
                    with gr.TabItem('模糊', elem_id='haku_blur'):
                        img3_h_slider = gr.Slider(160, 1280, 320, step=10, label="圖片預覽高度", elem_id='haku_img3_h')
                        image_main2 = gr.Image(type='numpy', label="圖1", elem_id='haku_img3', show_label=False)
                        blur_slider = gr.Slider(0, 32, 8, label="模糊強度")
                        blur_btn = gr.Button("refresh", variant="primary")
                        
                    with gr.TabItem('色彩', elem_id='haku_color'):
                        imgc_h_slider = gr.Slider(160, 1280, 320, step=10, label="圖片預覽高度", elem_id='haku_imgc_h')
                        image_color = gr.Image(type='numpy', label="圖1", elem_id='haku_imgc')
                        temp_slider = gr.Slider(-100, 100, 0, step=1, label="色溫")
                        hue_slider = gr.Slider(-90, 90, 0, step=1, label="色調")
                        bright_slider = gr.Slider(-100, 100, 0, step=1, label="亮度")
                        contrast_slider = gr.Slider(-100, 100, 0, step=1, label="對比度")
                        sat_slider = gr.Slider(-100, 100, 0, step=1, label="飽和度")
                        gamma_slider = gr.Slider(0.2, 2.2, 1, step=0.1, label="Gamma")
                        color_btn = gr.Button("refresh", variant="primary")
            
            with gr.Column():
                image_out = gr.Image(interactive=False, type='pil', label="haku_output", elem_id='haku_out')
                with gr.Row():
                    send_btns = gpc.create_buttons(["img2img", "inpaint", "extras"])
                with gr.Row():
                    send_first = gr.Button(">> 疊圖1", elem_id='send_haku_blend1')
                    send_second = gr.Button(">> 疊圖2", elem_id='send_haku_blend2')
                    send_blur = gr.Button(">> 模糊", elem_id='send_haku_blur')
                    send_color = gr.Button(">> 色彩", elem_id='send_haku_color')
        img1_h_slider.change(None, img1_h_slider, _js=f'get_change_height("#haku_img1")')
        img2_h_slider.change(None, img2_h_slider, _js=f'get_change_height("#haku_img2")')
        img3_h_slider.change(None, img3_h_slider, _js=f'get_change_height("#haku_img3")')
        imgc_h_slider.change(None, imgc_h_slider, _js=f'get_change_height("#haku_imgc")')
        
        # blend
        all_blend_input = [image_main, image_mask, alpha_slider, mask2_slider, mask2_blur_slider]
        image_main.change(on_change("img1"), image_main)
        image_mask.change(on_change("img2"), image_mask)
        
        alpha_slider.change(on_change("alpha", run), [alpha_slider]+all_blend_input, image_out)
        mask2_slider.change(on_change("m2_str", run), [mask2_slider]+all_blend_input, image_out)
        mask2_blur_slider.change(on_change("m2_blur", run), [mask2_blur_slider]+all_blend_input, image_out)
        expand_btn.click(run, all_blend_input, outputs=image_out)
        
        #blur
        all_blur_input = [image_main2, blur_slider]
        image_main2.change(on_change("img3"), image_main2)
        blur_slider.change(on_change("img_blur", blur), [blur_slider]+all_blur_input, image_out)
        blur_btn.click(blur, all_blur_input, outputs=image_out)
        
        #color
        all_color_input = [
            image_color, 
            bright_slider, contrast_slider, sat_slider, 
            temp_slider, hue_slider, gamma_slider
        ]
        image_color.change(on_change('imgc'), image_color, image_out)
        bright_slider.change(on_change('bright', run_color), [bright_slider]+all_color_input, image_out)
        contrast_slider.change(on_change('contrast', run_color), [contrast_slider]+all_color_input, image_out)
        sat_slider.change(on_change('sat', run_color), [sat_slider]+all_color_input, image_out)
        temp_slider.change(on_change('temp', run_color), [temp_slider]+all_color_input, image_out)
        hue_slider.change(on_change('hue', run_color), [hue_slider]+all_color_input, image_out)
        gamma_slider.change(on_change('gamma', run_color), [gamma_slider]+all_color_input, image_out)
        color_btn.click(run_color, all_color_input, outputs=image_out)
        
        for btn1, btn2, btn3, btn4, img_src in all_btns:
            print(img_src)
            btn1.click(gpc.image_from_url_text, img_src, image_main, _js="extract_image_from_gallery")
            btn2.click(gpc.image_from_url_text, img_src, image_mask, _js="extract_image_from_gallery")
            btn3.click(gpc.image_from_url_text, img_src, image_main2, _js="extract_image_from_gallery")
            btn4.click(gpc.image_from_url_text, img_src, image_color, _js="extract_image_from_gallery")
        
        gpc.bind_buttons(send_btns, image_out, None)
        send_first.click(lambda x:x, image_out, image_main)
        send_second.click(lambda x:x, image_out, image_mask)
        send_blur.click(lambda x:x, image_out, image_main2)
        send_color.click(lambda x:x, image_out, image_color)
        
        send_first.click(None, _js = 'switch_to_haku_blend')
        send_second.click(None, _js = 'switch_to_haku_blend')
        send_blur.click(None, _js = 'switch_to_haku_blur')
        send_color.click(None, _js = 'switch_to_haku_color')
    return (demo , "HakuImg", "haku_img"),


script_callbacks.on_ui_tabs(add_tab)