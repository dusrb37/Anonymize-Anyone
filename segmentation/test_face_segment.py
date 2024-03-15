import torch
import facer
from facer.util import bchw2hwc
from PIL import Image
import matplotlib.pyplot as plt

def covert_PIL_image(image: torch.Tensor):
    if image.dtype != torch.uint8:
        image = image.to(torch.uint8)
    if image.size(2) == 1:
        image = image.repeat(1, 1, 3)
    pimage = Image.fromarray(image.cpu().numpy())

    return pimage

def show_hwc(image: torch.Tensor):
    if image.dtype != torch.uint8:
        image = image.to(torch.uint8)
    if image.size(2) == 1:
        image = image.repeat(1, 1, 3)
    pimage = Image.fromarray(image.cpu().numpy())
    plt.imshow(pimage)
    plt.show()

def write_hwc(image: torch.Tensor, path: str):
    if image.dtype != torch.uint8:
        image = image.to(torch.uint8)
    if image.size(2) == 1:
        image = image.repeat(1, 1, 3)
    pimage = Image.fromarray(image.cpu().numpy())
    pimage = pimage.save(path)

def show_bchw(image: torch.Tensor):
    show_hwc(bchw2hwc(image))

def show_bhw(image: torch.Tensor):
    show_bchw(image.unsqueeze(1))

def write_bchw(image: torch.Tensor, path: str):
    write_hwc(bchw2hwc(image), path)

def write_bhw(image: torch.Tensor, path: str):
    write_bchw(image.unsqueeze(1), path)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

image = facer.hwc2bchw(facer.read_hwc('/segmentation/img/0.png')).to(device=device)  # image: 1 x 3 x h x w
face_detector = facer.face_detector('retinaface/mobilenet', device=device)

with torch.inference_mode():
    faces = face_detector(image)

face_parser = facer.face_parser('farl/celebm/448', device=device) 

with torch.inference_mode():
    faces = face_parser(image, faces)

seg_logits = faces['seg']['logits']
seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
n_classes = seg_probs.size(1)
vis_seg_probs = seg_probs.argmax(dim=1).float()/n_classes*255
vis_img = vis_seg_probs.sum(0, keepdim=True)

show_bchw(facer.draw_bchw(image, faces))

write_bchw(facer.draw_bchw(image, faces),'./segmentation/output/output.png')
