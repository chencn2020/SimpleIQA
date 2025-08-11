import torch
import torchvision

from SimpleIQA.utils.dataset import folders
from SimpleIQA.utils.dataset.process import ToTensor, Normalize, RandHorizontalFlip, Resize

class Data_Loader():
    """Dataset class for IQA databases"""

    def __init__(self, batch_size, dataset, path, img_indx, dataset_cfg, istrain=True, dist_type=None, **kwargs):
        self.resize_size = getattr(dataset_cfg, 'resize_size', (224, 224))
        self.num_workers = getattr(dataset_cfg, 'num_workers', 4)
        
        self.batch_size = batch_size
        self.istrain = istrain

        transforms = []
        if istrain and getattr(dataset_cfg, 'random_flipping', False):
            transforms.append(torchvision.transforms.RandomHorizontalFlip(p=getattr(dataset_cfg, 'random_flipping_rate', 0.5)))

        if getattr(dataset_cfg, 'resize_img', False):
            transforms.append(torchvision.transforms.Resize(size=self.resize_size))

        if getattr(dataset_cfg, 'random_crop', False):
            transforms.append(torchvision.transforms.RandomCrop(size=getattr(dataset_cfg, 'random_crop_size', (224, 224))))

        transforms.append(torchvision.transforms.ToTensor())
        transforms.append(
            torchvision.transforms.Normalize(
                mean=getattr(dataset_cfg, 'img_normalize_mean', (0.485, 0.456, 0.406)),
                std=getattr(dataset_cfg, 'img_normalize_std', (0.229, 0.224, 0.225))
            )
        )

        transforms=torchvision.transforms.Compose(transforms=transforms)
        
        if dataset == 'livec':
            self.data = folders.LIVEC(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain, dataset_cfg=dataset_cfg)
        elif dataset == 'koniq10k':
            self.data = folders.Koniq10k(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'uhdiqa':
            self.data = folders.uhdiqa(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'bid':
            self.data = folders.BID(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif 'spaq' in dataset:
            train_idx = int(dataset.split('_')[-1])
            self.data = folders.SPAQ(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain, column=train_idx)
        elif dataset == 'flive':
            self.data = folders.FLIVE(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'csiq':
            self.data = folders.CSIQ(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain, dist_type=dist_type)
        elif dataset == 'live':
            self.data = folders.LIVEFolder(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'kadid':
            self.data = folders.KADID(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'gfiqa_20k':
            self.data = folders.GFIQA_20k(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'agiqa_3k':
            self.data = folders.AGIQA_3k(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'aigciqa2023':
            self.data = folders.AIGCIQA2023(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'uwiqa':
            self.data = folders.UWIQA(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'cgiqa6k':
            self.data = folders.CGIQA6k(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        elif dataset == 'aigciqa3w':
            self.data = folders.AIGCIQA3W(root=path, index=img_indx, transform=transforms, batch_size=batch_size, istrain=istrain)
        else:
            raise NotImplementedError()
    
    def get_data(self):
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=self.istrain, num_workers=self.num_workers, drop_last=self.istrain)
        return dataloader
    
    def get_samples(self):
        return self.data
        
    def get_prompt(self, n=5, sample_type='fix'):
        prompt_data = self.data.get_promt(n=n, sample_type=sample_type)
        return torch.utils.data.DataLoader(prompt_data, batch_size=prompt_data.__len__(), shuffle=False)