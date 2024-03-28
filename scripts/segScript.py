import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn 
from scripts import modelDef
#import modelDef
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph
from skimage import filters
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes

model = modelDef.UNet(1, 64)

try:
  checkpoint = torch.load("torch_models/modelv4_currency.pt", map_location=torch.device('cpu'))
  #checkpoint = torch.load("../torch_models/modelv4_currency.pt", map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint['model_state_dict'])
except Exception as e:
  print(e)

def show_img(img):
    plt.imshow(img)
    plt.show()

def bounded_image(filePath, label, acc):
    test_img = Image.open(f"uploads/{filePath}")
    #test_img = Image.open("test_pic_3.jpg")

    transform_1 = transforms.Compose([transforms.Resize((512, 512)), transforms.Grayscale(), transforms.ToTensor()])
    transform_2 = transforms.Compose([transforms.Resize((512, 512)), transforms.PILToTensor()])

    t_test_img_1 = transform_1(test_img)
    t_test_img_2 = transform_2(test_img)

    out = model(t_test_img_1.unsqueeze(0))
    out_bin = (out > 0.5).float()

    out_nump = out_bin.squeeze(0).squeeze(0).numpy()

    selem_1 = morph.disk(radius=30)
    selem_2 = morph.disk(radius=30)
    morphed = morph.opening(morph.closing(out_nump, selem_1), selem_2)

    morphed_tensor = (torch.from_numpy(morphed)).bool()
    
    box = masks_to_boxes(morphed_tensor.unsqueeze(0))
    drawn_boxes = draw_bounding_boxes(t_test_img_2, box, colors="blue", width=5, 
                                      labels=[f"{label}, confidence: {acc:.2f}%"], 
                                      font="Mohave/static/Mohave-Medium.ttf", font_size=23)

    final_image = transforms.ToPILImage()(drawn_boxes)

    final_image.save("bounded_image.png", "PNG")
