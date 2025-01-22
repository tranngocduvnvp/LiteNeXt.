import torch
import torch.nn.functional as F

def dilatted(mask, kernel_size=3):
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    mask = F.conv2d(mask, kernel, padding="same")
    mask = torch.clip(mask, 0, 1)
    return mask

def erosin(mask, kernel_size=3):
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)*(1/(kernel_size**2))
    mask = F.conv2d(mask, kernel, padding="same")
    mask = torch.floor(mask + 1e-2)
    return mask

def margin(mask, kernel_size):
    mask_dilated = dilatted(mask, kernel_size)
    mask_erosin = erosin(mask, kernel_size)
    return mask_dilated - mask_erosin

def marginweight(mask, weight_in=3, weight_out=5, weight_margin=2, kernel_size=7):
    mask_dilated = dilatted(mask, kernel_size)
    mask_erosin = erosin(mask, kernel_size)
    return (mask_dilated - mask_erosin)*weight_margin + mask_erosin*weight_in + (1 - mask_dilated)*weight_out