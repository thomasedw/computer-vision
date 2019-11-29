from torchvision import models
import torchvision.transforms as transform
import numpy as np
import torch

class Segmenter():
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = models.segmentation.deeplabv3_resnet101(pretrained=True).eval().to(self.device)
        self.transforms = transform.Compose([
            transform.Resize(256),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.457, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def segment_image(self, image):
        network_input = self.transforms(image).unsqueeze(0)
        network_input = network_input.to(self.device)
        network_output = self.net(network_input)['out'].detach().cpu()
    
        output_map_encoded = torch.argmax(network_output.squeeze(), dim=0)
        output_map = self.decode_segmentation_map(output_map_encoded)

        return output_map
    
    def decode_segmentation_map(self, output_map, number_of_classes=21):
        label_colours = np.array([
            (0, 0, 0,), (128, 0, 0), (0, 128, 0),           # background, aeroplane, bicycle
            (128, 128, 0), (0, 0, 128), (128, 0, 128),      # bird, boat, bottle
            (0, 128, 128), (128, 128, 128), (64, 0, 0),     # bus, car, cat
            (192, 0, 0), (64, 128, 0), (192, 128, 0),       # chair, cow, dining table
            (64, 0, 128), (192, 0, 128), (64, 128, 128),    # cow, dining table, dog
            (192, 0, 128), (64, 128, 128), (192, 128, 128), # horse, motorbike, person
            (0, 64, 0), (128, 64, 0), (0, 192, 0),          # potted plant, sheep, sofa
            (128, 192, 0), (0, 64, 128)                     # train, tv/monitor
        ])

        r = np.zeros_like(output_map).astype(np.uint8)
        g = np.zeros_like(output_map).astype(np.uint8)
        b = np.zeros_like(output_map).astype(np.uint8)
        
        for l in range(0, number_of_classes):
            index = output_map == l
            r[index] = label_colours[l, 0]
            g[index] = label_colours[l, 1]
            b[index] = label_colours[l, 2]
            
        rgb = np.stack([r, g, b], axis=2)

        return rgb