from torch.utils.data.dataloader import DataLoader
from dataloader.ADEK_dataset import ADEKDataset


class ADEKDataLoader(DataLoader):
    def __init__(self, src_dir, batch_size=4, nproc=0,
                 target_dim=(256, 256)):
        self.train_dataset = ADEKDataset(src_dir, target_dim=target_dim)
        self.val_dataset = ADEKDataset(src_dir, target_dim=target_dim, is_train=False)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=nproc, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=nproc, pin_memory=True)

    def get_loader(self):
        return self.train_dataloader, self.val_dataloader


