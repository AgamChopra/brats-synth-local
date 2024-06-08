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

from numpy import array
from tqdm import trange
from matplotlib import pyplot as plt

import dataloader
import models
from utils import PSNR_Metric, SSIM_Metric, train_visualize, norm
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
          device='cpu', model_path=None, n=1, dropout=0.1,
          HYAK=False):

    print(device)

    # model
    neural_network = models.Attention_UNetT(in_c=2, out_c=1, n=n,
                                            dropout_rate=dropout).to(device)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        neural_network.load_state_dict(state_dict, strict=True)
    # neural_network = torch.compile(neural_network)

    # optimizer
    optimizer = torch.optim.AdamW(neural_network.parameters(), lr)

    # toptimization criterion
    criterion_masked = [Mask_MSELoss(), Mask_L1Loss()]
    criterion = [nn.MSELoss(), nn.L1Loss(),
                 ssim_loss(win_size=3, win_sigma=0.1),
                 GMELoss3D(device=device)]
    lambdas = [0.1, 0.15, 0.6, 0.3]

    # dataloader
    dataloader_paths = ['/gscratch/kurtlab/agam/data/brats-local-synthesis/TrainingData/', '/gscratch/kurtlab/agam/data/brats-local-synthesis/ValidationData/'] if HYAK else [
        '/home/agam/Desktop/brats_2024_local_impainting/TrainingData/', '/home/agam/Desktop/brats_2024_local_impainting/ValidationData/']
    data = dataloader.DataLoader(batch=batch, augment=True,
                                 aug_thresh=0.05, workers=4, norm=True,
                                 path=dataloader_paths[0])
    data_val = dataloader.DataLoader(batch=batch, augment=False,
                                     workers=4, norm=True,
                                     path=dataloader_paths[1])

    # how many times to iterate each epoch
    iterations = 1 * (int(data.max_id / batch) + (data.max_id % batch > 0))
    iterations_val = 1 * (int(data_val.max_id / batch) +
                          (data_val.max_id % batch > 0))

    # evaluation metrics and logging for visualization and evaluation
    ssim_metric = SSIM_Metric()
    psnr_metric = PSNR_Metric()
    mse_metric = nn.MSELoss()
    mae_metric = nn.L1Loss()

    losses, losses_train, losses_temp, losses_val = [], [], [], []
    mse, mae, psnr, ssim = [], [], [], []
    mse_val, mae_val, psnr_val, ssim_val = [], [], [], []

    scaler = GradScaler()

    # optimization loop
    for eps in range(epochs):
        print('Epoch %d:' % (eps))

        neural_network.train()
        # print(next(neural_network.parameters()).is_cuda)

        for _ in trange(iterations):
            optimizer.zero_grad()

            x, mask = data.load_batch()
            x, mask = x.float().to(device), mask.float().to(device)
            whole_mask = (x > 0.).float().to(device)

            known_masked_region = x * mask
            input_image = (x > 0) * ((mask < 0.5) * x)
            input_image = torch.cat((input_image, mask), dim=1)

            with autocast():
                y = whole_mask * neural_network(input_image).float()
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
        for _ in trange(3 * iterations_val):
            with torch.no_grad():
                x, mask = data.load_batch()
                x, mask = x.float().to(device), mask.float().to(device)
                whole_mask = (x > 0.).float().to(device)

                known_masked_region = x * mask
                input_image = (x > 0) * ((mask < 0.5) * x)
                input_image = torch.cat((input_image, mask), dim=1)

                with autocast():
                    y = whole_mask * neural_network(input_image).float()
                    synthetic_masked_region = mask * y

                    error = 10 * sum([efunc(known_masked_region,
                                            synthetic_masked_region,
                                            mask)
                                      for efunc in criterion_masked])

                    error += sum([lambdas[i] * criterion[i](
                        x, y) for i in range(len(criterion))])

                losses_temp.append(error.item())
                mse.append(-torch.log10(mse_metric(x, y) + 1e-12).item())
                mae.append(-torch.log10(mae_metric(x, y) + 1e-12).item())
                ssim.append(-torch.log10(1 -
                                         ssim_metric(x, y) + 1e-12).item())
                psnr.append(psnr_metric(x, y).item())

        mse_val.append(sum(mse)/len(mse))
        mae_val.append(sum(mae)/len(mae))
        ssim_val.append(sum(ssim)/len(ssim))
        psnr_val.append(sum(psnr)/len(psnr))

        losses_val.append(sum(losses_temp)/len(losses_temp))
        losses_temp = []
        mse, mae, psnr, ssim = [], [], [], []

        if (eps % 1 == 0 or eps == epochs - 1) and eps != 0:
            train_visualize([losses_train, losses_val,
                             mae_val, mse_val, ssim_val, psnr_val],
                            gans=False, dpi=180, HYAK=HYAK)

        if eps % 10 == 0 and eps != 0:
            torch.save(neural_network.state_dict(),
                       checkpoint_path + 'checkpoint_%d_epochs.pt' % (eps))

        if mse_val[-1] >= max(mse_val):
            torch.save(neural_network.state_dict(),
                       checkpoint_path + 'best_mse.pt')
            best_mse = eps

        if ssim_val[-1] >= max(ssim_val):
            torch.save(neural_network.state_dict(),
                       checkpoint_path + 'best_ssim.pt')
            best_ssim = eps

        if psnr_val[-1] >= max(psnr_val):
            torch.save(neural_network.state_dict(),
                       checkpoint_path + 'best_psnr.pt')
            best_psnr = eps

        norm_metrics = array([norm(mse_val), norm(ssim_val), norm(psnr_val)])
        avg_metrics = norm_metrics.sum(axis=0)

        if avg_metrics[-1] >= avg_metrics.max():
            torch.save(neural_network.state_dict(),
                       checkpoint_path + 'best_average.pt')
            best_avg = eps

        print('Average Train Loss: %.6f, Validation Loss: %.6f' %
              (losses_train[-1], losses_val[-1]))
        print('Validation MSE: %.6f, MAE: %.6f, PSNR: %.6f, SSIM: %.6f' %
              (mse_val[-1], mae_val[-1], psnr_val[-1], ssim_val[-1]))
        print('Best epochs for mse:%d, ssim:%d, psnr:%d, norm_average:%d' %
              (best_mse, best_ssim, best_psnr, best_avg))

    torch.save(neural_network.state_dict(),
               checkpoint_path + 'checkpoint_%d_epochs.pt' % (eps))

    return losses_train, losses_val


def validate(checkpoint_path, model_path, batch=1,
             n=1, epochs=100, dropout=0.1, device='cpu',
             HYAK=False):
    print(device)

    model = models.Attention_UNetT(in_c=2, out_c=1, n=n,
                                   dropout_rate=dropout).to(device)
    state_dict = torch.load(checkpoint_path + model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dataloader_path = '/gscratch/kurtlab/agam/data/brats-local-synthesis/ValidationData/' if HYAK else '/home/agam/Desktop/brats_2024_local_impainting/ValidationData/'
    data = dataloader.DataLoader(batch=batch, augment=False,
                                 workers=4, norm=True,
                                 path=dataloader_path)

    iterations = 1 * (int(data.max_id / batch) + (data.max_id % batch > 0))

    scores = [nn.MSELoss(), nn.L1Loss(), PSNR_Metric(), SSIM_Metric()]
    mse, mae, psnr, ssim = [], [], [], []

    with torch.no_grad():
        for eps in range(epochs):
            for _ in trange(iterations):
                x, mask = data.load_batch()
                x, mask = x.to(device).float(), mask.to(device).float()
                whole_mask = (x > 0.).float().to(device)

                with autocast():
                    input_image = (x > 0) * ((mask < 0.5) * x)
                    input_image = torch.cat((input_image, mask), dim=1)
                    y = torch.clip(
                        model(input_image), 0., 1.)
                    synthetic_masked_region = mask * y * whole_mask

                    score = [scores[i](
                        x,
                        x * (mask < 0.5) + synthetic_masked_region
                    ) for i in range(
                        len(scores))]

                mse.append(score[0].item())
                mae.append(score[1].item())
                psnr.append(score[2].item())
                ssim.append(score[3].item())

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
        print('Total Average MSE: %.4f, MAE: %.4f, PSNR: %.4f, SSIM: %.4f' %
              (sum(mse)/len(mse), sum(mae)/len(mae),
               sum(psnr)/len(psnr), sum(ssim)/len(ssim)))


def trn(checkpoint_path, epochs=500, lr=1E-4, batch=1,
        device='cpu', n=1, params=None, dropout=0.1, HYAK=False):
    losst, lossv = train(
        checkpoint_path=checkpoint_path,
        epochs=epochs, lr=lr, batch=batch,
        device=device, model_path=params, n=n,
        dropout=dropout, HYAK=HYAK
    )
    plt.plot(losst, label='Training Loss')
    plt.plot(lossv, label='Validation Loss')
    plt.title('Compound Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    HYAK = False
    checkpoint_path = '/gscratch/kurtlab/brats2024/repos/agam/brats-synth-local/log' if\
        HYAK else '/home/agam/Documents/git-files/brats-synth-local/'
    model_path = 'best_average.pt'
    fresh = False
    epochs = 500
    lr = 1E-4
    batch = 1
    device = 'cuda'
    n = 2
    dropout = 0

    # trn(checkpoint_path, epochs=epochs, lr=lr,
    #     batch=batch, device=device, n=n,
    #     params=None if fresh else checkpoint_path + model_path,
    #     dropout=dropout, HYAK=HYAK)

    validate(checkpoint_path, model_path=model_path, epochs=10,
             dropout=dropout, batch=batch, n=n, device=device, HYAK=HYAK)
