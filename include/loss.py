import torch 
import torch.nn as nn

def calc_covmat(batch_size, n_antenna, y_pred, min_angle, max_angle, resolution):
    COV_Mat_gen = torch.zeros((batch_size, n_antenna, n_antenna), dtype=torch.complex64)
    steering_vec = torch.zeros((n_antenna, 1), dtype=torch.complex64)
    angle_range = torch.arange(min_angle, max_angle+1, resolution)  # -60 ~ 60 (121)
        
    for batch_idx in range(batch_size):
        for idx, angle in enumerate(angle_range):
            rad = torch.deg2rad(angle)
            for antenna_idx in range(n_antenna):
                steering_vec[antenna_idx] = torch.exp(1j * torch.pi * torch.sin(rad) * antenna_idx) # (4, 1)
            COV_Mat_gen[batch_idx] += y_pred[batch_idx, idx] * (steering_vec @ steering_vec.conj().transpose(1, 0))

    return COV_Mat_gen

class radarLoss(nn.Module):
    def __init__(self, min_angle, max_angle, resolution, device):
        super(radarLoss, self).__init__()
        
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.resolution = resolution
        self.device = device
        
    def forward(self, y_pred, y_true):
        batch_size, n_channel, n_antenna, n_antenna = y_true.shape
        COV_Mat_gen = calc_covmat(batch_size, n_antenna, y_pred.cpu(), self.min_angle, self.max_angle, self.resolution)
        
        pred = torch.concatenate((torch.real(COV_Mat_gen), torch.imag(COV_Mat_gen), torch.angle(COV_Mat_gen)), axis=1).to(self.device)
        pred = pred.reshape(batch_size, n_channel, n_antenna, n_antenna)

        return torch.sqrt(nn.functional.mse_loss(y_true, pred))