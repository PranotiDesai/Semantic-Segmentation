import os
import cv2
import torch
from network.deep_lap_v3 import DeepLabV3Plus
from util.adek_colormap import ADEKColormap
from util.utills import load_model


class Test:
    def __init__(self, n_class, cuda_flag=False, target_dim=(256, 256)):
        self.cuda_flag = cuda_flag
        self.target_dim = target_dim
        self.model = DeepLabV3Plus(n_class, output_stride=16)
        self.colormap = ADEKColormap(n_class, cuda_flag)
        if cuda_flag:
            self.model = self.model.cuda()
        else:

            self.model = self.model.cpu()

    def run(self):
        self.model.eval()
        files = os.listdir("sample_images/images/")
        for file in files:
            image = cv2.cvtColor(cv2.imread("sample_images/images/"+file), cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()/255
            if self.cuda_flag:
                image = image.cuda()
            pred = torch.argmax(self.model(image), dim=1)
            smap = self.colormap(pred)
            smap = (smap.permute(0, 2, 3, 1)[0]*255).type(torch.uint8)
            smap = cv2.cvtColor(smap.cpu().numpy(), cv2.COLOR_BGR2RGB)
            cv2.imwrite("sample_images/predictions/%s.png" % file.split(".")[0], smap)


if __name__ == "__main__":
    test = Test(n_class=151, cuda_flag=True, target_dim=(512, 256))
    test.model = load_model(test.model, "weights/model.pth")
    test.run()
    print("Annotations are saved in sample_images/predictions")
