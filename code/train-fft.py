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
import argparse
from tqdm import trange
from numpy import array
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from matplotlib import pyplot as plt

import dataloader
import modelsT as models
from utils import PSNR_Metric, SSIM_Metric, SSIMLoss, GMELoss3D, norm, pad3d
from utils import get_fft_mag_phase, reconst_image, train_visualize

# Set PyTorch printing precision
torch.set_printoptions(precision=8)

# Set precision for matrix multiplication
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train(checkpoint_path, epochs=200, lr=1E-4, batch=1, device='cpu',
          model_path=[None, None], dropout=0.1, HYAK=False, gui=True):
    """
    Training function for the model.

    Args:
        checkpoint_path (str): Path to save checkpoints.
        epochs (int, optional): Number of epochs. Default is 200.
        lr (float, optional): Learning rate. Default is 1E-4.
        batch (int, optional): Batch size. Default is 1.
        device (str, optional): Device to run the training on. Default is 'cpu'.
        model_path ([str,str], optional): Path to the model checkpoint. Default is [None, None].
        n (int, optional): Scaling factor for the number of filters of upscaling model. Default is 1.
        dropout (float, optional): Dropout rate. Default is 0.1.
        HYAK (bool, optional): Flag to use HYAK paths. Default is False.
        gui (bool, optional): Flag to plot figures. Default is True.
    """
    print(device)

    # Initialize models
    fft_transformer_model = models.Transformer_Model(depth=64).to(device)
    upscale_model = models.Attention_UNet(
        2, 1, n=16, dropout_rate=dropout).to(device)
    if model_path[0] is not None:
        state_dict = torch.load(model_path[0], map_location=device)
        fft_transformer_model.load_state_dict(state_dict, strict=True)

        state_dict = torch.load(model_path[1], map_location=device)
        upscale_model.load_state_dict(state_dict, strict=True)

    # Initialize optimizers
    optimizer_fftT = torch.optim.AdamW(fft_transformer_model.parameters(), lr)
    optimizer_upscale = torch.optim.AdamW(upscale_model.parameters(), lr)

    # Initialize loss functions and criteria
    criterion_fft = [nn.L1Loss()]
    criterion_upscale = [nn.MSELoss(),
                         nn.L1Loss(),
                         SSIMLoss(win_size=3, win_sigma=0.1),
                         GMELoss3D(device=device)]

    # Initialize dataloaders
    dataloader_paths = [
        '/gscratch/kurtlab/agam/data/brats-local-synthesis/TrainingData/',
        '/gscratch/kurtlab/agam/data/brats-local-synthesis/ValidationData/'
    ] if HYAK else [
        '/home/agam/Desktop/brats_2024_local_impainting/TrainingData/',
        '/home/agam/Desktop/brats_2024_local_impainting/ValidationData/'
    ]
    data = dataloader.DataLoader(
        batch=batch, augment=True, aug_thresh=0.05, workers=4, norm=True, path=dataloader_paths[0])
    data_val = dataloader.DataLoader(
        batch=batch, augment=False, workers=4, norm=True, path=dataloader_paths[1])

    # Calculate number of iterations per epoch
    iterations = (data.max_id // batch) + (data.max_id % batch > 0)
    iterations_val = (data_val.max_id // batch) + (data_val.max_id % batch > 0)

    # Initialize evaluation metrics
    ssim_metric = SSIM_Metric()
    psnr_metric = PSNR_Metric()
    mse_metric = nn.MSELoss()
    mae_metric = nn.L1Loss()

    # Initialize lists for logging losses and metrics
    losses, losses_train, losses_temp, losses_val = [], [], [], []
    upscale_losses, upscale_losses_train = [], []
    mse, mae, psnr, ssim = [], [], [], []
    mse_val, mae_val, psnr_val, ssim_val = [], [], [], []

    scaler = GradScaler()

    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch}:')
        fft_transformer_model.train()
        upscale_model.train()

        for _ in trange(iterations):
            x, mask = data.load_batch()
            if x is None or mask is None:
                print("Train batch returned None values.")
                continue

            x, mask = pad3d(x, 240).float().to(
                device), pad3d(mask, 240).float().to(device)

            input_image = (x > 0) * ((mask < 0.5) * x)
            input_mag_phase = torch.cat((get_fft_mag_phase(input_image),
                                         get_fft_mag_phase(mask)), dim=1).detach()
            x_mag_phase = get_fft_mag_phase(x).detach()

            optimizer_fftT.zero_grad()
            optimizer_upscale.zero_grad()

            with autocast():
                y = fft_transformer_model(input_mag_phase).float()

                error = torch.nan_to_num(sum(
                    [efunc(x_mag_phase, y) for efunc in criterion_fft]), posinf=5E2, neginf=1E-6, nan=10)

                # print(error)

                scaler.scale(error).backward()
                scaler.step(optimizer_fftT)
                scaler.update()

                losses.append(error.item())

                upscale_input = torch.cat(
                    (reconst_image(y), input_image), dim=1).float().detach()

                y = upscale_model(upscale_input).float()

                error = torch.nan_to_num(sum([efunc(
                    x.detach(), y) for efunc in criterion_upscale]), posinf=5E2, neginf=1E-6, nan=10)

                # print(error)

                scaler.scale(error).backward()
                scaler.step(optimizer_upscale)
                scaler.update()

                upscale_losses.append(error.item())

        losses_train.append(sum(losses) / len(losses) if losses else 0)
        losses = []

        upscale_losses_train.append(
            sum(upscale_losses) / len(upscale_losses) if upscale_losses else 0)
        upscale_losses = []

        # Validation loop
        fft_transformer_model.eval()
        upscale_model.eval()

        for _ in trange(3 * iterations_val):
            with torch.no_grad():
                x, mask = data_val.load_batch()
                if x is None or mask is None:
                    print("Validation batch returned None values.")
                    continue

                x, mask = pad3d(x, 240).float().to(
                    device), pad3d(mask, 240).float().to(device)

                input_image = (x > 0) * ((mask < 0.5) * x)
                input_mag_phase = torch.cat((get_fft_mag_phase(input_image),
                                             get_fft_mag_phase(mask)), dim=1)

                y = fft_transformer_model(input_mag_phase).float()
                y = torch.cat((reconst_image(y), input_image), dim=1).float()
                y = upscale_model(y).float()

                error = sum([efunc(x, y) for efunc in criterion_upscale])

                losses_temp.append(error.item())
                mse.append(-torch.log10(mse_metric(x, y) + 1e-12).item())
                mae.append(-torch.log10(mae_metric(x, y) + 1e-12).item())
                ssim.append(-torch.log10(1 - ssim_metric(x, y) + 1e-12).item())
                psnr.append(psnr_metric(x, y).item())

        mse_val.append(sum(mse) / len(mse))
        mae_val.append(sum(mae) / len(mae))
        ssim_val.append(sum(ssim) / len(ssim))
        psnr_val.append(sum(psnr) / len(psnr))
        losses_val.append(sum(losses_temp) / len(losses_temp))
        losses_temp = []
        mse, mae, psnr, ssim = [], [], [], []

        if (epoch % 1 == 0 or epoch == epochs - 1) and epoch != 0 and gui:
            train_visualize([upscale_losses_train, losses_train, losses_val, mae_val,
                            mse_val, ssim_val, psnr_val], gans=True, dpi=180, HYAK=HYAK)

        if epoch % 10 == 0 and epoch != 0:
            torch.save(fft_transformer_model.state_dict(),
                       f"{checkpoint_path}fft_checkpoint_{epoch}_epochs.pt")
            torch.save(upscale_model.state_dict(),
                       f"{checkpoint_path}upscale_checkpoint_{epoch}_epochs.pt")

        if mse_val[-1] >= max(mse_val):
            torch.save(fft_transformer_model.state_dict(),
                       f"{checkpoint_path}fft_best_mse.pt")
            torch.save(upscale_model.state_dict(),
                       f"{checkpoint_path}upscale_best_mse.pt")
            best_mse = epoch

        if ssim_val[-1] >= max(ssim_val):
            torch.save(fft_transformer_model.state_dict(),
                       f"{checkpoint_path}fft_best_ssim.pt")
            torch.save(upscale_model.state_dict(),
                       f"{checkpoint_path}upscale_best_ssim.pt")
            best_ssim = epoch

        if psnr_val[-1] >= max(psnr_val):
            torch.save(fft_transformer_model.state_dict(),
                       f"{checkpoint_path}fft_best_psnr.pt")
            torch.save(upscale_model.state_dict(),
                       f"{checkpoint_path}upscale_best_psnr.pt")
            best_psnr = epoch

        norm_metrics = array([norm(mse_val), norm(ssim_val), norm(psnr_val)])
        avg_metrics = norm_metrics.sum(axis=0)

        if avg_metrics[-1] >= avg_metrics.max():
            torch.save(fft_transformer_model.state_dict(),
                       f"{checkpoint_path}fft_best_avg.pt")
            torch.save(upscale_model.state_dict(),
                       f"{checkpoint_path}upscale_best_avg.pt")
            best_avg = epoch

        print(
            f'Average Train Loss: {losses_train[-1]:.6f}, {upscale_losses_train[-1]:.6f}, Validation Loss: {losses_val[-1]:.6f}')
        print(
            f'Validation MSE: {mse_val[-1]:.6f}, MAE: {mae_val[-1]:.6f}, PSNR: {psnr_val[-1]:.6f}, SSIM: {ssim_val[-1]:.6f}')
        if epoch > 0:
            print(
                f'Best epochs for mse: {best_mse}, ssim: {best_ssim}, psnr: {best_psnr}, norm_average: {best_avg}')

    torch.save(fft_transformer_model.state_dict(),
               f"{checkpoint_path}fft_checkpoint_{epoch}_epochs.pt")
    torch.save(upscale_model.state_dict(),
               f"{checkpoint_path}upscale_checkpoint_{epoch}_epochs.pt")

    return losses_train, losses_val


def validate(checkpoint_path, model_path, batch=1, n=1, epochs=100,
             dropout=0.1, device='cpu', HYAK=False, gui=True):
    """
    Validation function for the model.

    Args:
        checkpoint_path (str): Path to save checkpoints.
        model_path (str): Path to the model checkpoint.
        batch (int, optional): Batch size. Default is 1.
        n (int, optional): Scaling factor for the number of channels. Default is 1.
        epochs (int, optional): Number of epochs. Default is 100.
        dropout (float, optional): Dropout rate. Default is 0.1.
        device (str, optional): Device to run the validation on. Default is 'cpu'.
        HYAK (bool, optional): Flag to use HYAK paths. Default is False.
    """
    fft_transformer_model = models.Transformer_Model(depth=64).to(device)
    upscale_model = models.Attention_UNet(
        2, 1, n=16, dropout_rate=dropout).to(device)

    state_dict = torch.load(model_path[0], map_location=device)
    fft_transformer_model.load_state_dict(state_dict, strict=True)
    state_dict = torch.load(model_path[1], map_location=device)
    upscale_model.load_state_dict(state_dict, strict=True)

    fft_transformer_model.eval()
    upscale_model.eval()

    dataloader_path = '/gscratch/kurtlab/agam/data/brats-local-synthesis/ValidationData/' if HYAK else '/home/agam/Desktop/brats_2024_local_impainting/ValidationData/'
    data = dataloader.DataLoader(
        batch=batch, augment=False, workers=4, norm=True, path=dataloader_path)

    iterations = (data.max_id // batch) + (data.max_id % batch > 0)

    scores = [nn.MSELoss(), nn.L1Loss(), PSNR_Metric(), SSIM_Metric()]
    mse, mae, psnr, ssim = [], [], [], []

    with torch.no_grad():
        for _ in range(epochs):
            for _ in trange(iterations):
                x, mask = data.load_batch()
                x, mask = pad3d(x, 240).float().to(
                    device), pad3d(mask, 240).float().to(device)
                whole_mask = (x > 0.).float().to(device)
                input_image = (x > 0) * ((mask < 0.5) * x)
                input_mag_phase = torch.cat((get_fft_mag_phase(input_image),
                                             get_fft_mag_phase(mask)), dim=1)

                with autocast():
                    y = fft_transformer_model(input_mag_phase).float()
                    y = torch.cat(
                        (reconst_image(y), input_image), dim=1).float()
                    y = upscale_model(y).float()
                    y = torch.clip(y, 0., 1.)
                    synthetic_masked_region = mask * y * whole_mask

                    score = [scores[i](
                        x, x * (mask < 0.5) + synthetic_masked_region) for i in range(len(scores))]

                mse.append(score[0].item())
                mae.append(score[1].item())
                psnr.append(score[2].item())
                ssim.append(score[3].item())

                if gui:
                    dataloader.show_images(
                        torch.cat(
                            (x.cpu(), input_image[:, 0:1].cpu(), (synthetic_masked_region + (x * (mask < 0.5))).cpu(),
                             y.cpu(), synthetic_masked_region.cpu(), torch.abs(x * mask - synthetic_masked_region).cpu()), dim=0), 6, 3, dpi=350)

        print(f'\n{model_path}')
        print(
            f'Total Average MSE: {sum(mse) / len(mse):.8f}, MAE: {sum(mae) / len(mae):.8f}, PSNR: {sum(psnr) / len(psnr):.8f}, SSIM: {sum(ssim) / len(ssim):.8f}')


def trn(checkpoint_path, epochs=500, lr=1E-4, batch=1, device='cpu', params=None, dropout=0.1, HYAK=False):
    """
    Wrapper function to train the model and plot the training and validation losses.

    Args:
        checkpoint_path (str): Path to save checkpoints.
        epochs (int, optional): Number of epochs. Default is 500.
        lr (float, optional): Learning rate. Default is 1E-4.
        batch (int, optional): Batch size. Default is 1.
        device (str, optional): Device to run the training on. Default is 'cpu'.
        n (int, optional): Scaling factor for the number of channels. Default is 1.
        params (list, optional): List of paths to model and critic checkpoints. Default is None.
        dropout (float, optional): Dropout rate. Default is 0.1.
        HYAK (bool, optional): Flag to use HYAK paths. Default is False.
    """
    losses_train, losses_val = train(
        checkpoint_path=checkpoint_path,
        epochs=epochs, lr=lr, batch=batch,
        device=device, model_path=params,
        dropout=dropout, HYAK=HYAK
    )
    plt.plot(losses_train, label='Training Loss')
    plt.plot(losses_val, label='Validation Loss')
    plt.title('Compound Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train with optional GUI.')
    parser.add_argument('--gui', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='Enable or disable GUI (default: True)')
    args = parser.parse_args()
    print(f"GUI Enabled: {args.gui}")

    HYAK = False
    checkpoint_path = '/gscratch/kurtlab/brats2024/repos/agam/brats-synth-local/log/' if HYAK else '/home/agam/Documents/git-files/brats-synth-local/param/'
    fft_path = 'fft_best_avg.pt'
    upscale_path = 'upscale_best_avg.pt'
    fresh = True
    epochs = 100
    lr = 1E-3
    batch = 1
    device = 'cuda'
    dropout = 0

    # trn(checkpoint_path, epochs=epochs, lr=lr, batch=batch, device=device,
    #     params=[None, None] if fresh else [f"{checkpoint_path}{fft_path}",
    #                                        f"{checkpoint_path}{upscale_path}"],
    #     dropout=dropout, HYAK=HYAK)

    # Uncomment to validate
    validate(checkpoint_path, model_path=[f"{checkpoint_path}{fft_path}",
                                          f"{checkpoint_path}{upscale_path}"], epochs=2,
             dropout=dropout, batch=batch, device=device, HYAK=HYAK, gui=args.gui)
