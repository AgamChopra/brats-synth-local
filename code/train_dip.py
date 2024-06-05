# -*- coding: utf-8 -*-
"""
Created on June 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
@Refs:
    - PyTorch 2.0 stable documentation @ https://pytorch.org/docs/stable/
"""
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torch.quantization as quant

from tqdm import trange
from matplotlib import pyplot as plt
from math import log10

import dataloader
import models
from utils import PSNR_Metric, SSIM_Metric, norm
from utils import ssim_loss, GMELoss3D, Mask_L1Loss, Mask_MSELoss

torch.set_printoptions(precision=8)

# 'highest', 'high', 'medium'. 'highest' is slower but accurate while 'medium'
#    is faster but less accurate. Before changing please refer to
#        https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision('high')

# 'True' = faster but less accurate, 'False' = Slower but more accurate
torch.backends.cuda.matmul.allow_tf32 = True

# 'True' = faster but less accurate, 'False' = Slower but more accurate
torch.backends.cudnn.allow_tf32 = True


def train(checkpoint_path, epochs=200, lr=1E-4, batch=1,
          device='cpu', model_path=None, n=1, dropout=0.1):

    print(device)

    # load the model
    neural_network = models.Attention_UNetT(in_c=2, out_c=1, n=n,
                                            dropout_rate=dropout).to(device)
    if model_path is not None:
        neural_network.load_state_dict(
            torch.load(model_path, map_location=device))
    # neural_network = torch.compile(neural_network)

    # load the optimizer and criterion
    optimizer = torch.optim.AdamW(neural_network.parameters(), lr)
    criterion_masked = [Mask_MSELoss(), Mask_L1Loss()]
    criterion = [ssim_loss(win_size=3, win_sigma=0.1),
                 GMELoss3D(device=device)]
    lambdas = [0.6, 0.3]

    # load dataloader
    data = dataloader.DataLoader(batch=batch, augment=True,
                                 aug_thresh=0.05, workers=8, norm=True,
                                 path='/home/agam/Desktop/brats_2024_local_impainting/TrainingData/')
    data_val = dataloader.DataLoader(batch=batch, augment=False,
                                     workers=4, norm=True,
                                     path='/home/agam/Desktop/brats_2024_local_impainting/ValidationData/')
    # print('%d training samples, %d validation samples' %
    #       (data.max_id, len(data_val.pid)))

    # how many times to iterate each epoch
    iterations = 1 * (int(data.max_id / batch) + (data.max_id % batch > 0))
    iterations_val = 1 * (int(data_val.max_id / batch) +
                          (data_val.max_id % batch > 0))

    # evaluation metrics and training loss for visualization and evaluation
    ssim_metric = SSIM_Metric()
    psnr_metric = PSNR_Metric()

    losses = []
    losses_train = []
    losses_temp = []
    losses_val = []
    mse_val = []
    mae_val = []
    ssim_val = []
    psnr_val = []
    mse = []
    mae = []
    ssim = []
    psnr = []

    scaler = GradScaler()

    # optimization loop
    for eps in range(epochs):
        print('Epoch %d:' % (eps))

        neural_network.train()
        # print(next(neural_network.parameters()).is_cuda)

        for i in trange(iterations):
            optimizer.zero_grad()

            x, mask = data.load_batch()
            x, mask = x.float().to(device), mask.float().to(device)
            known_masked_region = x * mask
            input_image = (x > 0) * ((mask < 0.5) * x)
            input_image = torch.cat((input_image, mask), dim=1)

            with autocast():
                y = neural_network(input_image).float()
                synthetic_masked_region = mask * y

                error = 10 * sum([efunc(known_masked_region,
                                        synthetic_masked_region,
                                        mask)
                                  for efunc in criterion_masked])

                error += sum([lambdas[i] * criterion[i](
                    x, y) for i in range(len(criterion))])

            scaler.scale(error).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(error.item())

        losses_train.append(sum(losses)/iterations)
        losses = []

        # validation loop (after each epoch)
        neural_network.eval()
        for i in trange(iterations_val):
            with torch.no_grad():
                x, mask = data_val.load_batch()
                x, mask = x.float().to(device), mask.float().to(device)

                input_image = (x > 0) * ((mask < 0.5) * x)
                input_image = torch.cat((input_image, mask), dim=1)

                with autocast():
                    y = neural_network(input_image).float()
                    error = [criterion[i](x, y) for i in range(len(criterion))]

                losses_temp.append(sum(error).item())
                mse.append(-torch.log10(error[0] + 1e-12).item())
                mae.append(-torch.log10(error[1] + 1e-12).item())
                ssim.append(-torch.log10(1 -
                                         ssim_metric(x, y) + 1e-12).item())
                psnr.append(psnr_metric(x, y).item())

        mse_val.append(sum(mse)/iterations_val)
        mae_val.append(sum(mae)/iterations_val)
        ssim_val.append(sum(ssim)/iterations_val)
        psnr_val.append(sum(psnr)/iterations_val)

        losses_val.append(sum(losses_temp)/iterations_val)
        losses_temp = []
        mse = []
        mae = []
        ssim = []
        psnr = []

        if (eps % 1 == 0 or eps == epochs - 1) and eps != 0:
            plt.figure(dpi=200)
            plt.plot(losses_train, label='Training Loss')
            plt.plot(losses_val, label='Validation Loss')
            plt.title('Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            plt.legend()
            plt.show()

            plt.figure(dpi=250)
            plt.plot(norm(psnr_val), label='PSNR')
            plt.plot(norm(ssim_val), label='-log 1-SSIM')
            plt.plot(norm(mse_val), label='-log MSE')
            plt.plot(norm(mae_val), label='-log MAE')
            plt.title('Normalized Validation Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('Normalized Metric Score')
            plt.legend()
            plt.show()

        if eps % 1 == 0 and eps != 0:
            torch.save(neural_network.state_dict(),
                       checkpoint_path + 'dip_checkpoint_%d_epochs.pt' % (eps))

        print(' Average Train Loss: %.8f, Validation Loss: %.8f' %
              (losses_train[-1], losses_val[-1]))

    torch.save(neural_network.state_dict(),
               checkpoint_path + 'dip_checkpoint_%d_epochs.pt' % (eps))

    return losses_train, losses_val


def quantize(checkpoint_path, model_path,
             batch=1, n=1, dropout=0.1):
    device = 'cpu'
    print(device)

    # load the model
    neural_network = models.Attention_UNetT(in_c=1, out_c=1, n=n,
                                            dropout_rate=dropout).to(device)
    neural_network.load_state_dict(torch.load(checkpoint_path + model_path,
                                              map_location=device))

    # calibration dataloader (subset of training dataloader...)
    calibration_data = dataloader.DataLoader(batch=batch, augment=False,
                                             workers=4, norm=True,
                                             path='/home/agam/Desktop/brats_2024_local_impainting/TrainingData/')

    # Prepare the model for quantization
    neural_network.eval()
    neural_network.qconfig = quant.QConfig(
        activation=quant.MinMaxObserver.with_args(dtype=torch.quint8),
        weight=quant.MinMaxObserver.with_args(dtype=torch.qint8)
    )
    quant.prepare(neural_network, inplace=True)

    # Calibration step (use a subset of your training data)
    for _ in trange(20):
        x, mask = calibration_data.load_batch()
        x = x.float().to(device)
        neural_network(x*(mask < 0.5))

    # Convert to a quantized model
    quantized_model = quant.convert(neural_network)

    quantized_model_path = checkpoint_path + 'quantized_' + model_path
    torch.save(quantized_model.state_dict(), quantized_model_path)


def validate(checkpoint_path, model_path, batch=1,
             n=1, epochs=100, dropout=0.1, device='cpu'):
    print(device)

    scores = [nn.MSELoss(), nn.L1Loss(), PSNR_Metric(), SSIM_Metric()]

    # load the model
    model = models.Attention_UNetT(in_c=2, out_c=1, n=n,
                                   dropout_rate=dropout).to(device)

    model.load_state_dict(torch.load(
        checkpoint_path + model_path, map_location=device))

    model.eval()

    data = dataloader.DataLoader(batch=batch, augment=False,
                                 workers=4, norm=True,
                                 path='/home/agam/Desktop/brats_2024_local_impainting/ValidationData/')

    iterations = 1 * (int(data.max_id / batch) +
                      (data.max_id % batch > 0))

    # evaluation scores
    mse = []
    mae = []
    psnr = []
    ssim = []

    with torch.no_grad():
        for eps in range(epochs):
            # print('Epoch %d:' % (eps))
            for i in trange(iterations):
                x, mask = data.load_batch()
                x, mask = x.to(device).float(), mask.to(device).float()

                with autocast():
                    input_image = (x > 0) * ((mask < 0.5) * x)
                    input_image = torch.cat((input_image, mask), dim=1)
                    y = torch.clip(
                        model(input_image), 0., 1.)
                    synthetic_masked_region = mask * y

                    score = [scores[i](
                        x,
                        x * (mask < 0.5) + synthetic_masked_region
                    ) for i in range(
                        len(scores))]

                mse.append(score[0].item())
                mae.append(score[1].item())
                psnr.append(score[2].item())
                ssim.append(score[3].item())

                # print('\nAverage MSE: %.8f, MAE: %.8f, PSNR: %.8f, SSIM: %.8f' %
                #       (sum(mse)/len(mse), sum(mae)/len(mae),
                #        sum(psnr)/len(psnr), sum(ssim)/len(ssim)))
                # print('Average -log(MSE): %.8f, -log(MAE): %.8f, PSNR: %.8f, -log(1-SSIM): %.8f' %
                #       (-log10(1e-12 + sum(mse)/len(mse)), -log10(1e-12 + sum(mae)/len(mae)),
                #        sum(psnr)/len(psnr), -log10(1e-12 + 1 - sum(ssim)/len(ssim))))

                dataloader.show_images(
                    torch.cat(
                        (x.cpu(),
                         input_image[:, 0:1].cpu(),
                         (synthetic_masked_region +
                          (x * (mask < 0.5))).cpu(),
                         y.cpu(),
                         synthetic_masked_region.cpu(),
                         torch.abs(x * mask - synthetic_masked_region).cpu()),
                        dim=0), 6, 3, dpi=350)

        print('\n', model_path)
        print('Total Average MSE: %.8f, MAE: %.8f, PSNR: %.8f, SSIM: %.8f' %
              (sum(mse)/len(mse), sum(mae)/len(mae),
               sum(psnr)/len(psnr), sum(ssim)/len(ssim)))
        print('Total Average -log(MSE): %.8f, -log(MAE): %.8f, PSNR: %.8f, -log(1-SSIM): %.8f' %
              (-log10(1e-12 + sum(mse)/len(mse)), -log10(1e-12 + sum(mae)/len(mae)),
               sum(psnr)/len(psnr), -log10(1e-12 + 1 - sum(ssim)/len(ssim))))


def trn(checkpoint_path, epochs=500, lr=1E-4, batch=1,
        device='cpu', n=1, params=None, dropout=0.1):
    losst, lossv = train(
        checkpoint_path=checkpoint_path,
        epochs=epochs, lr=lr, batch=batch,
        device=device, model_path=params, n=n,
        dropout=dropout
    )
    plt.plot(losst, label='Training Loss')
    plt.plot(lossv, label='Validation Loss')
    plt.title('Compound Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    checkpoint_path = '/home/agam/Documents/git-files/brats-synth-local/'
    # 'aunet_n4_lr1e-3_60eps_drp1_finetune_lr1e-4_98_eps.pt'
    model_path = 'dip_checkpoint_1_epochs.pt'
    fresh = True
    epochs = 100
    lr = 1E-3
    batch = 1
    device = 'cuda'
    n = 64
    dropout = 0

    # trn(checkpoint_path, epochs=epochs, lr=lr,
    #     batch=batch, device=device, n=n,
    #     params=None if fresh else checkpoint_path + model_path,
    #     dropout=dropout)

    # quantize(checkpoint_path, model_path=model_path, dropout=dropout,
    #         batch=batch, n=n)

    validate(checkpoint_path, model_path=model_path, epochs=5,
             dropout=dropout, batch=batch, n=n, device=device)
