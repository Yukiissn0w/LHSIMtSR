import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataloader == 'cityscape':
        from data.cityscape_dataset import CityscapeDataset
        dataset = CityscapeDataset()
    elif opt.dataloader == 'ade20k':
        from data.ade20k_dataset import ADE20KDataset
        dataset = ADE20KDataset()
    elif opt.dataloader == 'Hotels50k_mix':
        from data.Hotels50k_mix_dataset import Hotels50kmixDataset
        dataset = Hotels50kmixDataset

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
