import torch
from train import DeepLabV3Trainer
from util.utills import load_model

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    trainer = DeepLabV3Trainer(n_class=151, cuda_flag=True, batch_size=4, target_dim=(512, 256), amp=True)
    trainer.model = load_model(trainer.model, "weights/model.pth")
    trainer.train(hm_epoch=100)
    print(list(trainer.model.state_dict().keys()))
