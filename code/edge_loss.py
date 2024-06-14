"""
Created on Tue Jul 10 2022
Last Modified on Thu Apr 6 2023

@author: Agamdeep Chopra, achopra4@uw.edu
@affiliation: University of Washington, Seattle WA
@reference: Thevenot, A. (2022, February 17). Implement canny edge detection
            from scratch with PyTorch. Medium. Retrieved July 10, 2022, from
            https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed
"""
import numpy as np
import torch
import torch.nn as nn

EPSILON = 1E-6


def get_sobel_kernel3D(n1=1, n2=2, n3=2):
    '''
    Returns 3D Sobel kernels Sx, Sy, Sz, and diagonal kernels
    ex:
        Sx = [[[-n1, 0, n1],
               [-n2, 0, n2],
               [-n1, 0, n1]],
              [[-n2, 0, n2],
               [-n2*n3, 0, n2*n3],
               [-n2, 0, n2]],
              [[-n1, 0, n1],
               [-n2, 0, n2],
               [-n1, 0, n1]]]


    Parameters
    ----------
    n1 : int, optional
        kernel value 1. The default is 1.
    n2 : int, optional
        kernel value 2. The default is 2.
    n3 : int, optional
        kernel value 3. The default is 2.

    Returns
    -------
    list
        list of all the 3d sobel kernels.

    '''
    Sx = np.asarray([[[-n1, 0, n1], [-n2, 0, n2], [-n1, 0, n1]], [[-n2, 0, n2],
                    [-n3*n2, 0, n3*n2], [-n2, 0, n2]], [[-n1, 0, n1], [-n2, 0, n2], [-n1, 0, n1]]])
    Sy = np.asarray([[[-n1, -n2, -n1], [0, 0, 0], [n1, n2, n1]], [[-n2, -n3*n2, -n2],
                    [0, 0, 0], [n2, n3*n2, n2]], [[-n1, -n2, -n1], [0, 0, 0], [n1, n2, n1]]])
    Sz = np.asarray([[[-n1, -n2, -n1], [-n2, -n3*n2, -n2], [-n1, -n2, -n1]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[n1, n2, n1], [n2, n3*n2, n2], [n1, n2, n1]]])
    Sd11 = np.asarray([[[0, n1, n2], [-n1, 0, n1], [-n2, -n1, 0]], [[0, n2, n2*n3],
                      [-n2, 0, n2], [-n2*n3, -n2, 0]], [[0, n1, n2], [-n1, 0, n1], [-n2, -n1, 0]]])
    Sd12 = np.asarray([[[-n2, -n1, 0], [-n1, 0, n1], [0, n1, n2]], [[-n2*n3, -n2, 0],
                      [-n2, 0, n2], [0, n2, n2*n3]], [[-n2, -n1, 0], [-n1, 0, n1], [0, n1, n2]]])
    Sd21 = Sd11.T
    Sd22 = Sd12.T
    Sd31 = np.asarray([-S.T for S in Sd11.T])
    Sd32 = np.asarray([S.T for S in Sd12.T])

    return [Sx, Sy, Sz, Sd11, Sd12, Sd21, Sd22, Sd31, Sd32]


class GradEdge3D():
    '''
    Sobel edge detection algorithm compatible with PyTorch Autograd engine.
    '''

    def __init__(self, n1=1, n2=2, n3=2, device='cpu'):
        super(GradEdge3D, self).__init__()
        self.device = device
        k_sobel = 3
        S = get_sobel_kernel3D(n1, n2, n3)
        self.sobel_filters = []

        for s in S:
            sobel_filter = nn.Conv3d(in_channels=1, out_channels=1, stride=1,
                                     kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
            sobel_filter.weight.data = torch.from_numpy(
                s.astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
            sobel_filter = sobel_filter.to(self.device, dtype=torch.float32)
            self.sobel_filters.append(sobel_filter)

    def detect(self, img, a=1):
        '''
        Detect edges using Sobel operator for a 3d image

        Parameters
        ----------
        img : torch tensor
            3D torch tensor of shape (b, c, x, y, z).
        a : int, optional
            padding to be added, do not change unless necessary. The default is 1.

        Returns
        -------
        torch tensor
            tensor of gradient edges of shape (b, 1, x, y, z).

        '''
        pad = (a, a, a, a, a, a)
        B, C, H, W, D = img.shape

        img = nn.functional.pad(img, pad, mode='reflect')

        grad_mag = (1 / C) * torch.sum(torch.stack([torch.sum(torch.cat([s(img[:, c:c+1])for c in range(
            C)], dim=1) + EPSILON, dim=1) ** 2 for s in self.sobel_filters], dim=1) + EPSILON, dim=1) ** 0.5
        grad_mag = grad_mag[:, a:-a, a:-a, a:-a]

        return grad_mag.view(B, 1, H, W, D)
