from __future__ import annotations

from PIL import Image, ImageFilter, ImageColor
import numpy as np
import cv2


class Blend:
    @classmethod
    def method(cls, name):
        return getattr(cls, name)
    
    @staticmethod
    def normal(target, blend, opacity, *args):
        return target * opacity + blend * (1-opacity)
    
    @staticmethod
    def darken(target, blend, *args):
        return np.minimum(target, blend)
    
    @staticmethod
    def multiply(target, blend, *args):
        return target * blend
    
    @staticmethod
    def color_burn(target, blend, *args):
        return 1 - (1-target)/blend
    
    @staticmethod
    def linear_burn(target, blend, *args):
        return target+blend-1
    
    @staticmethod
    def lighten(target, blend, *args):
        return np.maximum(target, blend)
    
    @staticmethod
    def screen(target, blend, *args):
        return 1 - (1-target) * (1-blend)
    
    @staticmethod
    def color_dodge(target, blend, *args):
        return target/(1-blend)
    
    @staticmethod
    def linear_dodge(target, blend, *args):
        return target+blend
    
    @staticmethod
    def overlay(target, blend, *args):
        return  (target>0.5) * (1-(2-2*target)*(1-blend)) +\
                (target<=0.5) * (2*target*blend)
    
    @staticmethod
    def soft_light(target, blend, *args):
        return  (blend>0.5) * (1 - (1-target)*(1-(blend-0.5))) +\
                (blend<=0.5) * (target*(blend+0.5))
    
    @staticmethod
    def hard_light(target, blend, *args):
        return  (blend>0.5) * (1 - (1-target)*(2-2*blend)) +\
                (blend<=0.5) * (2*target*blend)
    
    @staticmethod
    def vivid_light(target, blend, *args):
        return  (blend>0.5) * (1 - (1-target)/(2*blend-1)) +\
                (blend<=0.5) * (target/(1-2*blend))
    
    @staticmethod
    def linear_light(target, blend, *args):
        return  (blend>0.5) * (target + 2*(blend-0.5)) +\
                (blend<=0.5) * (target + 2*blend)
    
    @staticmethod
    def pin_light(target, blend, *args):
        return  (blend>0.5) * np.maximum(target,2*(blend-0.5)) +\
                (blend<=0.5) * np.minimum(target,2*blend)
    
    @staticmethod
    def difference(target, blend, *args):
        return np.abs(target - blend)
    
    @staticmethod
    def exclusion(target, blend, *args):
        return 0.5 - 2*(target-0.5)*(blend-0.5)

blend_methods = [i for i in Blend.__dict__.keys() if i[0]!='_' and i!='method']


def run(layers):
    def blend(bg, *args):
        assert len(args)%5==0
        chunks = [args[i*layers: i*layers+layers] for i in range(5)]
        h, w, c = [i['image'] for i in chunks[-1] if i is not None][0].shape
        base_img = np.array(Image.new(mode="RGB", size=(w, h), color=ImageColor.getcolor(bg, 'RGB')))
        base_img = base_img.astype(np.float64)/255
        
        for alpha, mask_blur, mask_str, mode, img in reversed(zip(*chunks)):
            if img is None or img['image'] is None: continue
            img_now = Image.fromarray(img['image']).resize((w, h))
            mask = Image.fromarray(img['mask'][:,:,0], mode='L')
            
            img_now = np.array(img_now).astype(np.float64)/255
            mask = mask.resize((w, h)).filter(ImageFilter.GaussianBlur(mask_blur))
            mask = np.expand_dims(np.array(mask)*mask_str/255, 2)
            
            img_now = Blend.normal(base_img, img_now, mask)
            base_img = Blend.method(mode)(img_now, base_img, alpha)
            base_img = np.clip(base_img, 0, 1)
        base_img *= 255
        
        return Image.fromarray(base_img.astype(np.uint8), mode='RGB')
    return blend