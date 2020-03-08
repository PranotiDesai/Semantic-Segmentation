import os
import torch
from dataloader.ADEK_dataloader import ADEKDataLoader
from network.deep_lap_v3 import DeepLabV3Plus
from tensorboardX import SummaryWriter
from util.viz import SegmentationVisualizer
from util.loss import SemanticLoss
from util.utills import compute_class_weights
from apex import amp


class DeepLabV3Trainer:
    def __init__(self, n_class, cuda_flag=False, batch_size=8, target_dim=(256, 256), amp=False):
        self.cuda_flag = cuda_flag
        self.batch_size = batch_size
        self.n_class = n_class
        self.target_dim = target_dim
        self.amp = amp
        self.writer = SummaryWriter()
        self.segmentation_visualizer = SegmentationVisualizer(self.writer, n_class, cuda_flag)
        data_dir = os.path.join("..", "ADE20K", "ADEChallengeData2016","")
        class_weights = compute_class_weights(self.n_class, data_dir,
                                              cuda_flag=cuda_flag)

        self.loss_function = SemanticLoss(n_class, class_weights=class_weights, ignore_index=255, cuda_flag=cuda_flag)

        self.train_loader, self.val_loader = ADEKDataLoader(data_dir,batch_size=batch_size, target_dim=target_dim,
                                                            nproc=24).get_loader()

        self.model = DeepLabV3Plus(n_class, output_stride=16)
        if cuda_flag:
            self.model = self.model.cuda()
            self.loss_function = self.loss_function.cuda()
            self.segmentation_visualizer = self.segmentation_visualizer.cuda()
        else:

            self.model = self.model.cpu()

    def train(self, hm_epoch):
        lr = 0.0001
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if self.amp:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level='O1', loss_scale="dynamic")
        iter = 0
        epoch_pixel_acc = 0
        epoch_miou = 0
        self.model.train()
        for epoch in range(hm_epoch):
            if (epoch+1) % 10 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr/2
            epoch_loss = 0
            for i, data in enumerate(self.train_loader):
                iter += 1
                images = data['images']
                annotations = data['annotations']
                if self.cuda_flag:
                    images = images.cuda()
                    annotations = annotations.cuda()

                optimizer.zero_grad()
                pred = self.model(images)  # ['out']
                ce_loss, smoothness_loss = self.loss_function(pred, annotations)
                loss = 0.9*ce_loss + 0.1*smoothness_loss
                if self.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                epoch_loss += loss.item()*images.shape[0]

                self.writer.add_scalar("Batch/Cross Entropy Loss", ce_loss, iter)
                self.writer.add_scalar("Batch/Smoothness Loss", smoothness_loss, iter)
                self.writer.add_scalar("Batch/Loss", loss, iter)
                # self.writer.add_scalar("Batch/Smoothness Loss", smoothness_loss, iter)

                pred = torch.argmax(pred, dim=1)

                if iter % 1000 == 0:
                    self.segmentation_visualizer(images, annotations,  pred, iter)

                pred = pred.long().view(-1, 1)  # .detach().cpu()
                annotations = annotations.view(-1, 1)  #.cpu()
                pixel_ac = torch.sum(pred == annotations).float()/pred.nelement()
                epoch_pixel_acc += pixel_ac.item()
                miou = DeepLabV3Trainer.get_miou(pred, annotations).item()
                epoch_miou += miou
                self.writer.add_scalar("Batch/MIOU Loss", miou, iter)
                self.writer.add_scalar("Batch/Pixel Accuracy", pixel_ac, iter)

            epoch_loss /= len(self.train_loader.dataset)
            epoch_pixel_acc = epoch_pixel_acc / (len(self.train_loader))
            epoch_miou = epoch_miou / (len(self.train_loader))
            self.writer.add_scalar("Epoch/Training Loss", epoch_loss, epoch)
            self.writer.add_scalar("Epoch/Training MIOU", epoch_miou, epoch)
            self.writer.add_scalar("Epoch/Training Pixel Accuracy", epoch_pixel_acc, epoch)
            torch.save(self.model.state_dict(), "weights/model.pth")

            # perform validation ###
            for i, data in enumerate(self.val_loader):
                iter += 1
                images = data['images']
                annotations = data['annotations']
                self.model.eval()
                with torch.no_grad():
                    if self.cuda_flag:
                        images = images.cuda()
                        annotations = annotations.cuda()

                    pred = self.model(images) # ['out']
                    ce_loss, _ = self.loss_function(pred, annotations)
                    smoothness_loss = smoothness_loss
                    loss = ce_loss + smoothness_loss
                    epoch_loss += loss.item()*images.shape[0]
                    pred = torch.argmax(pred, dim=1)
                    pred = pred.long().view(-1, 1)  # .detach().cpu()
                    annotations = annotations.view(-1, 1)  #.cpu()
                    pixel_ac = torch.sum(pred == annotations).float()/pred.nelement()
                    epoch_pixel_acc += pixel_ac.item()
                    miou = DeepLabV3Trainer.get_miou(pred, annotations).item()
                    epoch_miou += miou

            epoch_loss /= len(self.val_loader.dataset)
            epoch_pixel_acc = epoch_pixel_acc / (len(self.val_loader))
            epoch_miou = epoch_miou / (len(self.val_loader))
            self.writer.add_scalar("Epoch/Validation Loss", epoch_loss, epoch)
            self.writer.add_scalar("Epoch/Validation MIOU", epoch_miou, epoch)
            self.writer.add_scalar("Epoch/Validation Pixel Accuracy", epoch_pixel_acc, epoch)

    @staticmethod
    def get_miou(p, t):
        intersection = (t & p).bool()
        union = (t | p).bool()
        return torch.sum(intersection).float() / torch.sum(union)

