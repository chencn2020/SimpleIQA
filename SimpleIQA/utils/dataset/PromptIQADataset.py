import torch.utils.data as data
from SimpleIQA.utils.dataset.loader_tools import *

class PromptIQADataset(data.Dataset):
    def __init__(self, dataset_name, transform, batch_size, istrain):
        super().__init__()
        self.dataset_name = dataset_name
        
        self.transform = transform
        self.batch_size = batch_size
        self.istrain = istrain
        
        self.samples_p, self.gt_p, self.samples, self.gt = None, None, None, None
        
    def reshuffle(self):
        shuffle_sample, shuffle_gt = reshuffle(self.samples_p, self.gt_p)
        self.samples, self.gt = split_array(shuffle_sample, self.batch_size), split_array(shuffle_gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
    
    def __getitem__(self, index):
        img_tensor, gt_tensor = get_item(self.samples, self.gt, index, self.transform, div=5)
        return img_tensor, gt_tensor, self.samples[index], self.dataset_name

    def __len__(self):
        length = len(self.samples)
        return length

    def get_promt(self, n=10, sample_type='fix'):
        return get_prompt(self.samples_p, self.gt_p, self.transform, n, self.__len__(), sample_type=sample_type)

    def _process_data(self, dataset_cfg, sample, gt, reverse_score=False):
        if getattr(dataset_cfg, 're_sample', False):
            sample = [sa for sa in sample for _ in range(getattr(dataset_cfg, 're_sample_times', 25)) ]
            gt = [g for g in gt for _ in range(getattr(dataset_cfg, 're_sample_times', 25)) ]

        if getattr(dataset_cfg, 'nornalize_score', False):
            gt = normalization(gt, reverse_score)
            
        self.samples_p, self.gt_p = sample, gt
        self.samples, self.gt = split_array(sample, self.batch_size), split_array(gt, self.batch_size)
        if len(self.samples[-1]) != self.batch_size and self.istrain:
            self.samples = self.samples[:-1]
            self.gt = self.gt[:-1]
        
    def _print_data_info(self):
        print(f"[Data Info] Loading {len(self.samples_p)} {'training' if self.istrain else 'testing'} samples from '{self.dataset_name}' dataset.")