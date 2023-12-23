import torch 
from torch.utils.data import Dataset

class RadarDataset(Dataset):
  def __init__(self, transform, data, label):
    super(RadarDataset, self).__init__()
    self.transform = transform
    self.label = self.transform(label)
    self.data = self.transform(data).type(torch.float)

  def __len__(self, ):
    return self.data.shape[0]

  def __getitem__(self, index):   
    return self.data[index], self.label[index]