import torch as th
import torch.nn.functional as F


def get_local_patches_kernel(kernel_size, device):
    patch_dim = kernel_size**2
    k = th.eye(patch_dim, device=device).view(patch_dim, 1, kernel_size, kernel_size)
    return k


def extract_local_patches(input, kernel, N=None, padding=0, stride=1):
    b, c, _, _ = input.size()
    if kernel is None:
        kernel = get_local_patches_kernel(kernel_size=N, device=input.device)
    flinput = input.flatten(0, 1).unsqueeze(1)
    patches = F.conv2d(flinput, kernel, padding=padding, stride=stride)
    _, _, h, w = patches.size()
    return patches.view(b, c, -1, h, w)
