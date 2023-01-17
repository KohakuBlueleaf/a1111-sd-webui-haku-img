from PIL import Image, ImageFilter, ImageColor
import numpy as np


def run(layers):
    def blend(bg, *args):
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
    return blend