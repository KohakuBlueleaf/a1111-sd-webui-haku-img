from modules import scripts
from modules import script_callbacks, shared
from modules import generation_parameters_copypaste as gpc
from modules.ui_components import FormRow

import gradio as gr

from PIL import Image, ImageFilter, ImageEnhance, ImageColor
import cv2
import numpy as np



'''
Functional part
'''

'''Blend'''
def run(bg, *args):
    assert len(args)%4==0
    chunks = [args[i*layers: i*layers+layers] for i in range(4)]
    h, w, c = [i['image'] for i in chunks[-1] if i is not None][0].shape
    base_img = np.array(Image.new(mode="RGB", size=(w, h), color=ImageColor.getcolor(bg, 'RGB')))
    base_img = base_img.astype(np.float64)
    
    for alpha, mask_blur, mask_str, img in zip(*chunks):
        if img is None or img['image'] is None: continue
        img_now = Image.fromarray(img['image']).resize((w, h))
        mask = Image.fromarray(img['mask'][:,:,0], mode='L')
        
        img_now = np.array(img_now).astype(np.float64)
        mask = mask.resize((w, h)).filter(ImageFilter.GaussianBlur(mask_blur))
        mask = np.expand_dims(np.array(mask)*mask_str/255, 2)
        
        img_now = base_img*mask + img_now*(1-mask)
        base_img = base_img*alpha + img_now*(1-alpha)
    return Image.fromarray(base_img.astype(np.uint8), mode='RGB')


'''Blur'''
def blur(img3, img_blur):
    img = Image.fromarray(img3)
    blur = ImageFilter.GaussianBlur(img_blur)
    return img.filter(blur)


'''Color'''
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


'''sketch'''
def fix_float(val, eps=1e-3):
    return float(val)-eps

def dog_filter(img, kernel=0, sigma=1.4, k_sigma=1.6, gamma=1):
    g1 = cv2.GaussianBlur(img, (kernel,kernel), sigma)
    g2 = cv2.GaussianBlur(img, (kernel,kernel), sigma*k_sigma)
    return g1 - fix_float(gamma) * g2

def xdog(img, sigma, k_sigma, eps, phi, gamma, color, scale=True):
    if color=='gray':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    dog = dog_filter(img, 0, sigma, k_sigma, gamma)
    dog = dog/dog.max()
    e = 1+np.tanh(fix_float(phi) * (dog-fix_float(eps)))
    e[e>=1] = 1
    
    if color=='gray':
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    if scale:
        return Image.fromarray((e*255).astype('uint8'))
    else:
        return Image.fromarray(e.astype('uint8')*255)


'''
UI part
'''
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
        
        val = kwargs.get("value", "")
        id = kwargs.get("elem_id")
        if 'extras' in str(val).lower() or 'extra' in str(id).lower():
            print('component val: ', val, kwargs.get("elem_id"))
        
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

all_btns: list[tuple[gr.Button, ...]] = []
layers = 5

def add_tab():
    print('add tab')
    with gr.Blocks(analytics_enabled=False) as demo:
        with FormRow().style(equal_height=False):
            with gr.Column():
                with gr.Tabs(elem_id="haku_blend_tabs"):
                    with gr.TabItem('疊圖', elem_id='haku_blend'):
                        all_layers = []
                        all_alphas = []
                        all_mask_str = []
                        all_mask_blur = []
                        with gr.Tabs(elem_id="haku_blend_layers_tabs"):
                            img_blend_h_slider = gr.Slider(160, 1280, 320, step=10, label="圖片預覽高度", elem_id='haku_img_h_blend')
                            for i in range(1, layers+1):
                                with gr.TabItem(f'圖層{i}', elem_id=f'haku_blend_layer{i}'):
                                    all_layers.append(
                                        gr.ImageMask(type='numpy', label=f"圖層{i}", elem_id=f'haku_img_blend{i}')
                                    )
                                    all_alphas.append(
                                        gr.Slider(0, 1, 0.5, label=f"圖層{i}透明度")
                                    )
                                    all_mask_blur.append(
                                        gr.Slider(0, 32, 4, label=f"圖層{i}蒙版模糊")
                                    )
                                    all_mask_str.append(
                                        gr.Slider(0, 1, 1, label=f"圖層{i}蒙版強度")
                                    )
                        bg_color = gr.ColorPicker('#FFFFFF', label='背景顏色')
                        expand_btn = gr.Button("refresh", variant="primary")
                    
                    with gr.TabItem('效果', elem_id='haku_eff'):
                        img_eff_h_slider = gr.Slider(160, 1280, 320, step=10, label="圖片預覽高度", elem_id='haku_img_h_eff')
                        image_eff = gr.Image(type='numpy', label="圖", elem_id='haku_img_eff', show_label=False)
                        with gr.Tabs(elem_id='effect_tabs'):
                            with gr.TabItem('色彩', elem_id='haku_color'):
                                temp_slider = gr.Slider(-100, 100, 0, step=1, label="色溫")
                                hue_slider = gr.Slider(-90, 90, 0, step=1, label="色調")
                                bright_slider = gr.Slider(-100, 100, 0, step=1, label="亮度")
                                contrast_slider = gr.Slider(-100, 100, 0, step=1, label="對比度")
                                sat_slider = gr.Slider(-100, 100, 0, step=1, label="飽和度")
                                gamma_slider = gr.Slider(0.2, 2.2, 1, step=0.1, label="Gamma")
                                color_btn = gr.Button("refresh", variant="primary")
                            
                            with gr.TabItem('模糊', elem_id='haku_blur'):
                                blur_slider = gr.Slider(0, 32, 8, label="模糊強度")
                                blur_btn = gr.Button("refresh", variant="primary")
                            
                            with gr.TabItem('線稿', elem_id='haku_sketch'):
                                sk_sigma = gr.Slider(1, 5, 1.4, step=0.05, label='sigma')
                                sk_k_sigma = gr.Slider(1, 5, 1.6, step=0.05, label='k_sigma')
                                sk_eps = gr.Slider(-0.2, 0.2, -0.03, step=0.005, label='epsilon')
                                sk_phi = gr.Slider(1, 50, 10, step=1, label='phi')
                                sk_gamma = gr.Slider(0.75, 1, 1, step=0.005, label='gamma')
                                sk_color = gr.Radio(['gray', 'rgb'], value='gray', label='color mode')
                                sk_scale = gr.Checkbox(False, label='use scale')
                                sketch_btn = gr.Button("refresh", variant="primary")
            
            with gr.Column():
                image_out = gr.Image(
                    interactive=False, 
                    type='pil', 
                    label="haku_output", 
                    elem_id='haku_out'
                )
                with gr.Row():
                    send_btns = gpc.create_buttons(["img2img", "inpaint", "extras"])
                with gr.Row():
                    with gr.Accordion('>> 疊圖'):
                        send_blends = []
                        for i in range(1, layers+1):
                            send_blends.append(gr.Button(f">> 圖層{i}", elem_id=f'send_haku_blend{i}'))
                    send_eff = gr.Button(">> 效果", elem_id='send_haku_blur')
        img_blend_h_slider.change(None, img_blend_h_slider, _js=f'get_change_height("haku_img_blend")')
        img_eff_h_slider.change(None, img_eff_h_slider, _js=f'get_change_height("haku_img_eff")')
        
        # blend
        all_blend_set = [bg_color]
        all_blend_set += all_alphas+all_mask_blur+all_mask_str
        all_blend_input = all_blend_set + all_layers
        for component in all_blend_set:
            component.change(run, all_blend_input, image_out)
        expand_btn.click(run, all_blend_input, image_out)
        
        #blur
        all_blur_input = [image_eff, blur_slider]
        blur_slider.change(blur, all_blur_input, outputs=image_out)
        blur_btn.click(blur, all_blur_input, outputs=image_out)
        
        #color
        all_color_set = [
            bright_slider, contrast_slider, sat_slider, 
            temp_slider, hue_slider, gamma_slider
            
        ]
        all_color_input = [image_eff] + all_color_set
        for component in all_color_set:
            component.change(run_color, all_color_input, image_out)
        color_btn.click(run_color, all_color_input, image_out)
        
        #sketch
        all_sk_set = [
            sk_sigma, sk_k_sigma, sk_eps, sk_phi, sk_gamma, sk_color, sk_scale
        ]
        all_sk_input = [image_eff] + all_sk_set
        for component in all_sk_set:
            component.change(xdog, all_sk_input, image_out)
        sketch_btn.click(xdog, all_sk_input, image_out)
        
        for btns, btn3, img_src in all_btns:
            for btn, img in zip(btns, all_layers):
                btn.click(gpc.image_from_url_text, img_src, img, _js="extract_image_from_gallery")
            btn3.click(gpc.image_from_url_text, img_src, image_eff, _js="extract_image_from_gallery")
        
        
        gpc.bind_buttons(send_btns, image_out, None)
        for btn, img in zip(btns, all_layers):
            btn.click(lambda x:x, image_out, img)
            btn.click(None, _js = 'switch_to_haku_blend')
            
        send_eff.click(lambda x:x, image_out, image_eff)
        send_eff.click(None, _js = 'switch_to_haku_eff')
        
    return (demo , "HakuImg", "haku_img"),


script_callbacks.on_ui_tabs(add_tab)