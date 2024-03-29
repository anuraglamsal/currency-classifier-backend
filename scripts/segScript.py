import torch
from torchvision import transforms
from PIL import Image
from scripts import modelDef
import matplotlib.pyplot as plt
import skimage.morphology as morph
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import masks_to_boxes

model = modelDef.UNet(1, 64)

try:
  checkpoint = torch.load("torch_models/modelv4_currency.pt", map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint['model_state_dict'])
except Exception as e:
  print(e)

def show_img(img):
    plt.imshow(img)
    plt.show()

def bounded_image(filePath, label, acc):
    img = Image.open(f"uploads/{filePath}")
    
    # for the model
    transform_1 = transforms.Compose([transforms.Resize((512, 512)), transforms.Grayscale(), transforms.ToTensor()])
    # for the picture to draw on
    transform_2 = transforms.Compose([transforms.Resize((512, 512)), transforms.PILToTensor()])

    t_img_1 = transform_1(img)
    t_img_2 = transform_2(img)

    out = model(t_img_1.unsqueeze(0))
    out_bin = (out > 0.5).float() 

    out_nump = out_bin.squeeze(0).squeeze(0).numpy()

    selem_1 = morph.disk(radius=30)
    selem_2 = morph.disk(radius=30)
    morphed = morph.opening(morph.closing(out_nump, selem_1), selem_2)

    morphed_tensor = (torch.from_numpy(morphed)).bool()
    
    box = masks_to_boxes(morphed_tensor.unsqueeze(0))
    drawn_boxes = draw_bounding_boxes(t_img_2, box, colors="blue", width=5, 
                                      labels=[f"{label}, confidence: {acc:.2f}%"], 
                                      font="Mohave/static/Mohave-Medium.ttf", font_size=23)

    final_image = transforms.ToPILImage()(drawn_boxes)

    final_image.save("bounded_image.png", "PNG")
