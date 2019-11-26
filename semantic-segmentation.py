from torchvision import models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

fcn = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

img = Image.open('./TTBB-durham-02-10-17-sub10/left-images/1506943047.478101_L.png')

trf = T.Compose([
    T.Resize(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inp = trf(img).unsqueeze(0)

out = fcn(inp)['out']

print(out.shape)

om = torch.argmax(out.squeeze(), dim=0).detach().cpu()

def decode_segmap(image, nc=21):
   
  label_colors = np.array([
                (0, 0, 0),       #0 = background
                (128, 0, 0),     #1 = aeroplane
                (0, 128, 0),     #2 = bicycle
                (128, 128, 0),   #3 = bird
                (0, 0, 128),     #4 = boat
                (128, 0, 128),   #5 = bottle
                (0, 128, 128),   #6 = bus
                (128, 128, 128), #7 = car
                (64, 0, 0),      #8 = cat
                (192, 0, 0),     #9 = chair
                (64, 128, 0),    #10 = cow
                (192, 128, 0),   #11 = dining table
                (64, 0, 128),    #12 = dog
                (192, 0, 128),   #13 = horse
                (64, 128, 128),  #14 = motorbike
                (192, 128, 128), #15 = person
                (0, 64, 0),      #16 = potted plant
                (128, 64, 0),    #17 = sheep
                (0, 192, 0),     #18 = sofa
                (128, 192, 0),   #19 = train
                (0, 64, 128)     #20 = tv/monitor
            ])
 
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
   
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
     
  rgb = np.stack([r, g, b], axis=2)
  return rgb

rgb = decode_segmap(om)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img)
ax2.imshow(rgb)

plt.show()