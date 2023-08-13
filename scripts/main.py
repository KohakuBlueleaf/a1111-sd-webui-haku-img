from __future__ import annotations
from typing import Any, Tuple, List, Union

from modules import shared
from modules import scripts
from modules import script_callbacks
from modules import generation_parameters_copypaste as gpc
from modules.ui_components import FormRow

import gradio as gr

from hakuimg import(
    blend,
    blur,
    color,
    sketch,
    pixel,
    neon,
    curve,
    chromatic,
    lens_distortion,
    custom_exif,
    tilt_shift,
    flip,
)
from inoutpaint import main as outpaint


'''
UI part
'''

inpaint_base: gr.Image
inpaint_mask: gr.Image
all_btns: List[Tuple[gr.Button, ...]] = []
layers = int(shared.opts.data.get('hakuimg_layer_num', 5))
points = int(shared.opts.data.get('hakuimg_curve_points', 3))


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

        if id in {'extras_tab', 'txt2img_send_to_extras', 'img2img_send_to_extras'}:
            with gr.Accordion('HakuImg', open=False):
                with gr.Column():
                    with gr.Accordion('Send to Blend', open=False):
                        btns = []
                        for i in range(layers, 0, -1):
                            btn1 = gr.Button(f"Send to Layer{i}")
                            btn1.click(None, _js=f"switch_to_haku_img")
                            btns.append(btn1)
                with gr.Column():
                    btn3 = gr.Button("Send to Effect")
                    btn3.click(None, _js=f"switch_to_haku_img_eff")

            all_btns.append((btns, btn3, img_src))

    def ui(self, is_img2img):
        return []


def add_tab():
    print('add tab')
    with gr.Blocks(analytics_enabled=False) as demo:
        with FormRow(equal_height=False):
            with gr.Column():
                with gr.Tabs(elem_id="haku_blend_tabs"):
                    with gr.TabItem('Blend', elem_id='haku_blend'):
                        all_layers = []
                        all_alphas = []
                        all_mask_str = []
                        all_mask_blur = []
                        all_mode = []
                        img_blend_h_slider = gr.Slider(160, 1280, 320, step=10, label="Image preview height", elem_id='haku_img_h_blend')
                        with gr.Tabs(elem_id="haku_blend_layers_tabs"):
                            for i in range(layers, 0, -1):
                                with gr.TabItem(f'Layer{i}', elem_id=f'haku_blend_Layer{i}'):
                                    all_layers.append(
                                        gr.ImageMask(type='numpy', label=f"Layer{i}", elem_id=f'haku_img_blend{i}')
                                    )
                                    all_alphas.append(
                                        gr.Slider(0, 1, 1, label=f"Layer{i} opacity")
                                    )
                                    all_mask_blur.append(
                                        gr.Slider(0, 32, 4, label=f"Layer{i} mask blur")
                                    )
                                    all_mask_str.append(
                                        gr.Slider(0, 1, 1, label=f"Layer{i} mask strength")
                                    )
                                    all_mode.append(
                                        gr.Dropdown(blend.blend_methods, value='normal', label='Blend mode')
                                    )
                        bg_color = gr.ColorPicker('#FFFFFF', label='background color')
                        expand_btn = gr.Button("refresh", variant="primary")

                    with gr.TabItem('Effect', elem_id='haku_eff'):
                        img_eff_h_slider = gr.Slider(160, 1280, 320, step=10, label="Image preview height", elem_id='haku_img_h_eff')
                        image_eff = gr.Image(type='numpy', label="img", elem_id='haku_img_eff', show_label=False)
                        with gr.Tabs(elem_id='effect_tabs'):
                            with gr.TabItem('Color', elem_id='haku_color'):
                                with gr.Row():
                                    temp_slider = gr.Slider(-100, 100, 0, step=1, label="temperature")
                                    hue_slider = gr.Slider(-90, 90, 0, step=1, label="hue")
                                with gr.Row():
                                    bright_slider = gr.Slider(-100, 100, 0, step=1, label="brightness")
                                    contrast_slider = gr.Slider(-100, 100, 0, step=1, label="contrast")
                                with gr.Row():
                                    sat_slider = gr.Slider(-100, 100, 0, step=1, label="saturation")
                                    gamma_slider = gr.Slider(0.2, 2.2, 1, step=0.1, label="Gamma")
                                with gr.Row():
                                    exposure_offset_slider = gr.Slider(0, 1, 0, label="ExposureOffset")
                                    vignette_slider = gr.Slider(0, 1, 0, label="Vignette")
                                with gr.Row():
                                    noise_slider = gr.Slider(0, 1, 0, label="Noise")
                                    sharpness_slider = gr.Slider(0, 1, 0, label="Sharpness")
                                with gr.Row():
                                    hdr_slider = gr.Slider(0, 1, 0, label="HDR")
                                with gr.Row():
                                    color_btn = gr.Button("refresh", variant="primary")
                                    color_rst_btn = gr.Button("reset")

                            with gr.TabItem('Tone Curve', elem_id='haku_curve'):
                                all_points = [[], [], [], []]
                                all_curve_defaults = [[], [], [], []]
                                all_curves = []
                                with gr.Tabs(elem_id='curve'):
                                    for index, tab in enumerate(['All', 'R', 'G', 'B']):
                                        with gr.TabItem(tab):
                                            for i in range(1, points+1):
                                                with gr.Row():
                                                    all_points[index] += [
                                                        gr.Slider(
                                                            0, 255, int(255*i/(points+1)),
                                                            step=1, label=f'point{i} x'
                                                        ),
                                                        gr.Slider(
                                                            0, 255, int(255*i/(points+1)),
                                                            step=1, label=f'point{i} y'
                                                        )
                                                    ]
                                                    all_curve_defaults[index] += [int(255*i/(points+1))]*2
                                            all_curves.append(gr.Image())
                                with gr.Row():
                                    curve_btn = gr.Button("refresh", variant="primary")
                                    curve_rst_btn = gr.Button("reset")

                            with gr.TabItem('Blur', elem_id='haku_blur'):
                                blur_slider = gr.Slider(0, 128, 8, label="blur")
                                blur_btn = gr.Button("refresh", variant="primary")

                            with gr.TabItem('Sketch', elem_id='haku_sketch'):
                                sk_kernel = gr.Slider(0, 25, 0, step=1, label='kernel size')
                                with gr.Row():
                                    sk_sigma = gr.Slider(1, 5, 1.4, step=0.05, label='sigma')
                                    sk_k_sigma = gr.Slider(1, 5, 1.6, step=0.05, label='k_sigma')

                                with gr.Row():
                                    sk_eps = gr.Slider(-0.2, 0.2, -0.03, step=0.005, label='epsilon')
                                    sk_phi = gr.Slider(1, 50, 10, step=1, label='phi')
                                    sk_gamma = gr.Slider(0.75, 1, 1, step=0.005, label='gamma')

                                sk_color = gr.Radio(['gray', 'rgb'], value='gray', label='color mode')
                                sk_scale = gr.Checkbox(False, label='use scale')
                                with gr.Row():
                                    sketch_btn = gr.Button("refresh", variant="primary")
                                    sketch_rst_btn = gr.Button("reset")

                            with gr.TabItem('Pixelize', elem_id='haku_Pixelize'):
                                p_colors = gr.Slider(2, 256, 128, step=1, label='colors')
                                p_dot_size = gr.Slider(1, 32, 6, step=1, label='dot size')
                                p_outline = gr.Slider(0, 10, 1, step=1, label='outline inflating')
                                p_smooth = gr.Slider(0, 10, 4, step=1, label='Smoothing')
                                p_mode = gr.Radio(
                                    ['kmeans', 'dithering', 'kmeans with dithering'],
                                    value='kmeans', label='Color reduce algo'
                                )
                                with gr.Row():
                                    pixel_btn = gr.Button("refresh", variant="primary")
                                    pixel_rst_btn = gr.Button("reset")

                            with gr.TabItem('Glow', elem_id='haku_Glow'):
                                neon_mode = gr.Radio(['BS', 'BMBL'], value='BS', label='Glow mode')
                                neon_blur = gr.Slider(2, 128, 16, step=1, label='range')
                                neon_str = gr.Slider(0, 1, 1, step=0.05, label='strength')
                                with gr.Row():
                                    neon_btn = gr.Button("refresh", variant="primary")
                                    neon_rst_btn = gr.Button("reset")

                            with gr.TabItem('Chromatic', elem_id='haku_Chromatic'):
                                chromatic_slider = gr.Slider(0, 3, 1, label="chromatic")
                                chromatic_blur = gr.Checkbox(label="Blur", value=False)
                                chromatic_btn = gr.Button("refresh", variant="primary")

                            with gr.TabItem("Lens distortion (Fisheye)", elem_id="haku_LensDistortion"):
                                lens_distortion_k1_slider = gr.Slider(
                                    -1, 1, 0,
                                    label="Concavity of distortion of circles",
                                )
                                lens_distortion_k2_slider = gr.Slider(
                                    -1, 1, 0,
                                    label="Amplification of distortion of circles",
                                )
                                lens_distortion_btn = gr.Button("refresh", variant="primary")

                            with gr.TabItem("Tilt shift", elem_id="haku_TiltShift"):
                                tilt_shift_focus_ratio = gr.Slider(-3, 3, 0, step=0.5, label="Positioning the effect on the y-axis")
                                tilt_shift_dof = gr.Slider(10, 100, 60, step=1, label="The width of the focus region in pixels")
                                tilt_shift_btn = gr.Button("refresh", variant="primary")

                    with gr.TabItem('Other'):
                        img_other_h_slider = gr.Slider(160, 1280, 320, step=10, label="Image preview height", elem_id='haku_img_h_oth')
                        image_other = gr.Image(type='numpy', label="img", elem_id='haku_img_other', show_label=False)
                        with gr.Tabs(elem_id='function list'):
                            with gr.TabItem('InOutPaint'):
                                iop_u = gr.Slider(0, 512, 0, step=64, label='fill up')
                                iop_d = gr.Slider(0, 512, 0, step=64, label='fill down')
                                iop_l = gr.Slider(0, 512, 0, step=64, label='fill left')
                                iop_r = gr.Slider(0, 512, 0, step=64, label='fill right')
                                iop_btn = gr.Button("refresh", variant="primary")
                            with gr.TabItem("Flip"):
                                flip_axis = gr.Radio(["horizontal", "vertical"], value="horizontal", label="Axis")
                                flip_btn = gr.Button("refresh", variant="primary")
                            with gr.TabItem("Custom EXIF"):
                                custom_exif_area = gr.TextArea(label="Custom parameters")
                                custom_exif_btn = gr.Button("refresh", variant="primary")

            with gr.Column():
                img_out_h_slider = gr.Slider(160, 1280, 420, step=10, label="Image preview height", elem_id='haku_img_h_out')
                res_info = gr.Textbox(label='Resolution')
                image_out = gr.Image(
                    interactive=False,
                    type='pil',
                    label="haku_output",
                    elem_id='haku_out'
                )
                image_mask = gr.Image(visible=False)
                with gr.Row():
                    send_btns = gpc.create_buttons(["img2img", "inpaint", "extras"])
                    send_ip_b = gr.Button("Send to inpaint upload", elem_id='send_inpaint_base')
                with gr.Row():
                    with gr.Accordion('Send to Blend', open=False):
                        send_blends = []
                        for i in range(layers, 0, -1):
                            send_blends.append(gr.Button(f"Send to Layer{i}", elem_id=f'send_haku_blend{i}'))
                    send_eff = gr.Button("Send to Effect", elem_id='send_haku_blur')

        #preview height slider
        img_blend_h_slider.change(None, img_blend_h_slider, _js=f'get_change_height("haku_img_blend")')
        img_eff_h_slider.change(None, img_eff_h_slider, _js=f'get_change_height("haku_img_eff")')
        img_other_h_slider.change(None, img_other_h_slider, _js=f'get_change_height("haku_img_other")')
        img_out_h_slider.change(None, img_out_h_slider, _js=f'get_change_height("haku_out")')
        image_out.change(lambda x:f'{x.width} x {x.height}', image_out, res_info)
        image_out.change(None, img_out_h_slider, _js=f'get_change_height("haku_out")')

        # blend
        all_blend_set = [bg_color]
        all_blend_set += all_alphas+all_mask_blur+all_mask_str+all_mode
        all_blend_input = all_blend_set + all_layers
        for component in all_blend_set:
            _release_if_possible(component, blend.run(layers), all_blend_input, image_out)
        expand_btn.click(blend.run(layers), all_blend_input, image_out)

        #blur
        all_blur_input = [image_eff, blur_slider]
        _release_if_possible(blur_slider, blur.run, all_blur_input, outputs=image_out)
        blur_btn.click(blur.run, all_blur_input, outputs=image_out)

        #chromatic
        all_chromatic_set = [chromatic_slider, chromatic_blur]
        all_chromatic_input = [image_eff] + all_chromatic_set
        for component in all_chromatic_set:
            _release_if_possible(component, chromatic.run, all_chromatic_input, image_out)
        chromatic_btn.click(chromatic.run, all_chromatic_input, image_out)

        #color
        all_color_set = [
            bright_slider, contrast_slider, sat_slider,
            temp_slider, hue_slider, gamma_slider,
            exposure_offset_slider, hdr_slider, noise_slider,
            sharpness_slider, vignette_slider
        ]
        all_color_input = [image_eff] + all_color_set
        for component in all_color_set:
            _release_if_possible(component, color.run, all_color_input, image_out)
        color_btn.click(color.run, all_color_input, image_out)
        color_rst_btn.click(lambda:[0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0], None, all_color_set)

        #curve
        all_curve_set = sum(all_points, start=[])
        all_curve_defaults = sum(all_curve_defaults, [])
        all_curve_input = [image_eff] + all_curve_set
        for index, components in enumerate(all_points):
            for component in components:
                _release_if_possible(component, curve.curve_img, components, all_curves[index])
            curve_btn.click(curve.curve_img, components, all_curves[index])
        curve_btn.click(curve.run(points), all_curve_input, image_out)
        curve_rst_btn.click(lambda: all_curve_defaults, None, all_curve_set)

        #sketch
        all_sk_set = [
            sk_kernel, sk_sigma, sk_k_sigma, sk_eps, sk_phi, sk_gamma, sk_color, sk_scale
        ]
        all_sk_input = [image_eff] + all_sk_set
        for component in all_sk_set:
            _release_if_possible(component, sketch.run, all_sk_input, image_out)
        sketch_btn.click(sketch.run, all_sk_input, image_out)
        sketch_rst_btn.click(lambda: [0, 1.4, 1.6, -0.03, 10, 1, 'gray', False], None, all_sk_set)

        #pixelize
        all_p_set = [
            p_colors, p_dot_size, p_smooth, p_outline, p_mode
        ]
        all_p_input = [image_eff] + all_p_set
        for component in all_p_set:
            _release_if_possible(component, pixel.run, all_p_input, image_out)
        pixel_btn.click(pixel.run, all_p_input, image_out)
        pixel_rst_btn.click(lambda: [16, 8, 0, 5, 'kmeans'], None, all_p_set)

        #neon
        all_neon_set = [
            neon_blur, neon_str, neon_mode,
        ]
        all_neon_input = [image_eff] + all_neon_set
        for component in all_neon_set:
            _release_if_possible(component, neon.run, all_neon_input, image_out)
        neon_btn.click(neon.run, all_neon_input, image_out)
        neon_rst_btn.click(lambda: [16, 1, 'BS'], None, all_neon_set)

        #lens distortion
        all_ = [
            lens_distortion_k1_slider,
            lens_distortion_k2_slider,
        ]
        input_ = [image_eff] + all_
        for component in all_:
            _release_if_possible(component, lens_distortion.run, input_, image_out)
        lens_distortion_btn.click(lens_distortion.run, input_, image_out)

        #tilt shift
        all_ = [tilt_shift_focus_ratio, tilt_shift_dof]
        input_ = [image_eff] + all_
        for component in all_:
            _release_if_possible(component, tilt_shift.run, input_, image_out)
        tilt_shift_btn.click(tilt_shift.run, input_, image_out)

        #iop
        all_iop_set = [
            iop_u, iop_d, iop_l, iop_r
        ]
        all_iop_input = [image_other] + all_iop_set
        for component in all_iop_set:
            _release_if_possible(component, outpaint.run, all_iop_input, [image_out, image_mask])
        iop_btn.click(outpaint.run, all_iop_input, [image_out, image_mask])

        #flip axis
        all_ = [flip_axis]
        input_ = [image_other] + all_
        for component in all_:
            _release_if_possible(component, flip.run, input_, image_out)
        flip_btn.click(flip.run, input_, image_out)

        #custom exif
        all_ = [custom_exif_area]
        input_ = [image_other] + all_
        custom_exif_btn.click(custom_exif.run, input_, image_out)

        #send
        print(all_btns)
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

        send_ip_b.click(lambda *x:x, [image_out, image_mask], [inpaint_base, inpaint_mask])
        send_ip_b.click(None, _js = 'switch_to_inpaint_upload')

        send_eff.click(lambda x:x, image_out, image_eff)
        send_eff.click(None, _js = 'switch_to_haku_eff')

    return (demo , "HakuImg", "haku_img"),


def _release_if_possible(component, *args, **kwargs):
    if isinstance(component, gr.events.Releaseable):
        component.release(*args, **kwargs)
    else:
        component.change(*args, **kwargs)


def on_ui_settings():
    section = ('haku-img', "HakuImg")
    shared.opts.add_option(
        "hakuimg_layer_num",
        shared.OptionInfo(
            5,
            "Total num of layers (reload required)",
            section=section
        )
    )
    shared.opts.add_option(
        "hakuimg_curve_points",
        shared.OptionInfo(
            3,
            "Total num of point for curve (reload required)",
            section=section
        )
    )


script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_ui_settings(on_ui_settings)
