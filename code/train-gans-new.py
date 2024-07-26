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
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from numpy import array
from tqdm import trange

import dataloader
import models
from utils import PSNR_Metric, SSIM_Metric, train_visualize, norm
from utils import SSIMLoss, GMELoss3D, Mask_L1Loss, Mask_MSELoss
from utils import grad_penalty

# Set PyTorch printing precision
torch.set_printoptions(precision=8)

# Set precision for matrix multiplication
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train(checkpoint_path, epochs=200, lr=1E-4, batch=1,
          device='cpu', model_path=None, critic_path=None, n=1,
          beta1=0.5, beta2=0.999, Lambda_penalty=10, dropout=0.1,
          HYAK=False, identity='', device1='cpu', device2='cpu'):
    """
    Training function for the model.

    Args:
        checkpoint_path (str): Path to save checkpoints.
        epochs (int, optional): Number of epochs. Default is 200.
        lr (float, optional): Learning rate. Default is 1E-4.
        batch (int, optional): Batch size. Default is 1.
        device (str, optional): Device to run the training on. Default is 'cpu'.
        model_path (str, optional): Path to the model checkpoint. Default is None.
        critic_path (str, optional): Path to the critic checkpoint. Default is None.
        n (int, optional): Scaling factor for the number of model filters. Default is 1.
        beta1 (float, optional): Beta1 for Adam optimizer. Default is 0.5.
        beta2 (float, optional): Beta2 for Adam optimizer. Default is 0.999.
        Lambda_penalty (int, optional): Lambda for gradient penalty. Default is 10.
        dropout (float, optional): Dropout rate. Default is 0.1.
        HYAK (bool, optional): Flag to use HYAK paths. Default is False.
    """
    print(device)

    # Initialize models
    neural_network = models.Global_UNet(
        in_c=1,
        out_c=1,
        embed_dim=512,
        n_heads=8,
        mlp_ratio=32,
        qkv_bias=True,
        dropout_rate=0.,
        mask_downsample=40,
        noise=True,
        device1=device1,
        device2=device2
        )
    print(
        f'Gen. size: {models.count_parameters(neural_network)/1000000}M')
    if model_path is not None:
        state_dict = torch.load(model_path)#, map_location=device)
        neural_network.load_state_dict(state_dict, strict=True)

    critic = models.CriticA(in_c=2, fact=8).to(device2)
    print(
        f'Crit. size: {models.count_parameters(neural_network)/1000000}M')
    if critic_path is not None:
        try:
            state_dict = torch.load(critic_path, map_location=device2)
            critic.load_state_dict(state_dict, strict=True)
        except Exception:
            print(f"{critic_path} not found!")

    # Initialize optimizers
    optimizer = torch.optim.AdamW(neural_network.parameters(), lr)
    optimizerC = torch.optim.AdamW(
        critic.parameters(), lr, betas=(beta1, beta2))

    # Initialize loss functions and criteria
    criterion_masked = [Mask_MSELoss(), Mask_L1Loss()]
    criterion = [SSIMLoss(win_size=3, win_sigma=0.1),
                 GMELoss3D(device=device),
                 nn.L1Loss()]
    lambdas = [0.5, 0.25, 0.25]

    # Initialize dataloaders
    dataloader_paths = [
        '/gscratch/kurtlab/agam/data/brats-local-synthesis/TrainingData/',
        '/gscratch/kurtlab/agam/data/brats-local-synthesis/ValidationData/'
    ] if HYAK else [
        '/home/agam/Desktop/brats_2024_local_impainting/TrainingData/',
        '/home/agam/Desktop/brats_2024_local_impainting/ValidationData/'
    ]
    data = dataloader.DataLoader(
        batch=batch, augment=True, aug_thresh=0.05, workers=4,
        norm=True, path=dataloader_paths[0])
    data_val = dataloader.DataLoader(
        batch=batch, augment=False, workers=4,
        norm=True, path=dataloader_paths[1])

    # Calculate number of iterations per epoch
    iterations = (data.max_id // batch) + (
        data.max_id % batch > 0)
    iterations_val = (data_val.max_id // batch) + (
        data_val.max_id % batch > 0)

    # Initialize evaluation metrics
    ssim_metric = SSIM_Metric()
    psnr_metric = PSNR_Metric()
    mse_metric = nn.MSELoss()
    mae_metric = nn.L1Loss()

    # Initialize lists for logging losses and metrics
    losses, losses_train, losses_temp, losses_val = [], [], [], []
    critic_losses, critic_losses_train = [], []
    mse, mae, psnr, ssim = [], [], [], []
    mse_val, mae_val, psnr_val, ssim_val = [], [], [], []

    scaler = GradScaler()

    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch}:')
        neural_network.train()
        critic.train()

        for itervar in trange(iterations):
            x, mask = data.load_batch()
            if x is None or mask is None:
                print("Batch returned None values.")
                continue

            x, mask = x.float().to(device), mask.float().to(device)
            whole_mask = (x > 0.).float().to(device)

            known_masked_region = x * mask
            input_image = (x > 0) * ((mask < 0.5) * x)
            input_image = torch.cat((input_image, mask), dim=1)

            if (itervar + 1) % 5 == 0:
                # Generator Optimization
                optimizer.zero_grad()
                with autocast():
                    y = whole_mask * \
                        neural_network(input_image).float()
                    synthetic_masked_region = mask * y

                    errorA = sum([efunc(known_masked_region,
                                        synthetic_masked_region, mask)
                                  for efunc in criterion_masked])
                    errorB = sum([lambdas[i] * criterion[i](x, y)
                                  for i in range(len(criterion))])
                    errorC = critic(torch.cat((input_image,
                                               synthetic_masked_region),
                                              dim=1).float()).mean()

                    error = 10 * errorA + 10 * errorB - 0.5 * errorC

                scaler.scale(error).backward()
                scaler.step(optimizer)
                scaler.update()

                losses.append(errorC.item())

            else:
                # Critic Optimization
                optimizerC.zero_grad()
                with autocast():
                    y = whole_mask.detach() * neural_network(
                        input_image).float()
                    synthetic_masked_region = mask * y

                    real_x = torch.cat(
                        (input_image, known_masked_region), dim=1).float()
                    fake_x = torch.cat(
                        (input_image, synthetic_masked_region.detach()),
                        dim=1).float()

                    error_fake = critic(fake_x).mean()
                    error_real = critic(real_x).mean()
                    penalty = grad_penalty(
                        critic, real_x, fake_x, Lambda_penalty)

                    error = error_fake - error_real + penalty

                scaler.scale(error).backward(retain_graph=True)
                scaler.step(optimizerC)
                scaler.update()

                critic_losses.append(error.item())

        losses_train.append(sum(losses) / len(losses))
        losses = []

        critic_losses_train.append(
            sum(critic_losses) / len(critic_losses))
        critic_losses = []

        print(
            f'Average Train Loss: gen:{losses_train[-1]:.6f} crit:{critic_losses_train[-1]:.6f}')

        # Validation loop
        neural_network.eval()
        for _ in trange(iterations_val):
            with torch.no_grad():
                x, mask = data_val.load_batch()
                x, mask = x.float().to(device), mask.float().to(device)
                whole_mask = (x > 0.).float().to(device)

                known_masked_region = x * mask
                input_image = (x > 0) * ((mask < 0.5) * x)
                input_image = torch.cat((input_image, mask), dim=1)

                with autocast():
                    y = whole_mask * \
                        neural_network(input_image.detach()).detach().float()
                    synthetic_masked_region = mask * y

                    error = 10 * \
                        sum([efunc(known_masked_region,
                                   synthetic_masked_region, mask)
                            for efunc in criterion_masked])
                    error += sum([lambdas[i] * criterion[i](x, y)
                                 for i in range(len(criterion))])

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

        if (epoch % 1 == 0 or epoch == epochs - 1) and epoch != 0:
            train_visualize([critic_losses_train, losses_train,
                             losses_val, mae_val,
                            mse_val, ssim_val, psnr_val],
                            gans=True, dpi=180,
                            path=checkpoint_path if HYAK else None,
                            identity=identity)

        if epoch % 10 == 0 and epoch != 0:
            torch.save(neural_network.state_dict(),
                       f"{checkpoint_path}checkpoint_{epoch}_epochs_{identity}.pt")
            torch.save(critic.state_dict(),
                       f"{checkpoint_path}critic_checkpoint_{epoch}_epochs_{identity}.pt")

        if mse_val[-1] >= max(mse_val):
            torch.save(neural_network.state_dict(),
                       f"{checkpoint_path}best_mse_{identity}.pt")
            best_mse = epoch

        if ssim_val[-1] >= max(ssim_val):
            torch.save(neural_network.state_dict(),
                       f"{checkpoint_path}best_ssim_{identity}.pt")
            best_ssim = epoch

        if psnr_val[-1] >= max(psnr_val):
            torch.save(neural_network.state_dict(),
                       f"{checkpoint_path}best_psnr_{identity}.pt")
            best_psnr = epoch

        norm_metrics = array([norm(mse_val), norm(ssim_val), norm(psnr_val)])
        avg_metrics = norm_metrics.sum(axis=0)

        if avg_metrics[-1] >= avg_metrics.max():
            torch.save(neural_network.state_dict(),
                       f"{checkpoint_path}best_average_{identity}.pt")
            best_avg = epoch

        print(f'Average Validation Loss: {losses_val[-1]:.6f}')
        print(
            f'Validation MSE: {mse_val[-1]:.6f}, MAE: {mae_val[-1]:.6f}, PSNR: {psnr_val[-1]:.6f}, SSIM: {ssim_val[-1]:.6f}')
        print(
            f'Best epochs for mse: {best_mse}, ssim: {best_ssim}, psnr: {best_psnr}, norm_average: {best_avg}')

    torch.save(neural_network.state_dict(),
               f"{checkpoint_path}checkpoint_{epochs}_epochs_{identity}.pt")

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
    with torch.no_grad():
        model = models.Attention_UNet(
            in_c=2, out_c=1, n=n, dropout_rate=dropout).to(device)
        state_dict = torch.load(
            f"{checkpoint_path}{model_path}", map_location=device)
        print(f"{checkpoint_path}{model_path}", device)
        model.load_state_dict(state_dict)
        model.eval()

        dataloader_path = '/gscratch/kurtlab/agam/data/brats-local-synthesis/ValidationData/' if HYAK else '/home/agam/Desktop/brats_2024_local_impainting/ValidationData/'
        data = dataloader.DataLoader(
            batch=batch, augment=False, workers=4, norm=True, path=dataloader_path)

        iterations = (data.max_id // batch) + (data.max_id % batch > 0)

        scores = [nn.MSELoss(), nn.L1Loss(), PSNR_Metric(), SSIM_Metric()]
        mse, mae, psnr, ssim = [], [], [], []

        for _ in range(epochs):
            for _ in trange(iterations):
                x, mask = data.load_batch()
                x, mask = x.to(device).float(), mask.to(device).float()
                whole_mask = (x > 0.).float().to(device)

                with autocast():
                    input_image = (x > 0) * ((mask < 0.5) * x)
                    input_image = torch.cat((input_image, mask), dim=1)
                    y = torch.clip(model(input_image), 0., 1.)
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
                            (x.cpu(), input_image[:, 0:1].cpu(),
                             (synthetic_masked_region + (x * (mask < 0.5))).cpu(),
                             y.cpu(), synthetic_masked_region.cpu(),
                             torch.abs(x * mask - synthetic_masked_region).cpu()),
                            dim=0), 6, 3, dpi=350)

        print(f'\n{model_path}')
        print(
            f'Total Average MSE: {sum(mse) / len(mse):.8f}, MAE: {sum(mae) / len(mae):.8f}, PSNR: {sum(psnr) / len(psnr):.8f}, SSIM: {sum(ssim) / len(ssim):.8f}')


def trn(checkpoint_path, epochs=500, lr=1E-4, batch=1, device='cpu', n=1,
        params=None, dropout=0.1, HYAK=False, identity=''):
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
        device=device, model_path=params[0],
        critic_path=params[1], n=n,
        dropout=dropout, HYAK=HYAK,
        identity=identity
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GANs.')
    parser.add_argument('--gui', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='Enable or disable GUI (default: True)')
    parser.add_argument('--hyak', type=lambda x: (str(x).lower() == 'true'),
                        default=False, help='Enable or disable HYAK mode (default: False)')
    parser.add_argument('--identity', default='gans',
                        help='identity of the run (default: gans)')
    args = parser.parse_args()
    print(f"GUI Enabled: {args.gui}")
    print(f"HYAK Enabled: {args.hyak}")
    print(f"Identity: {args.identity}")

    HYAK = args.hyak
    checkpoint_path = '/gscratch/kurtlab/brats2024/repos/agam/brats-synth-local/log/' if HYAK else '/home/agam/Desktop/hyak-current-log/'
    model_path = 'checkpoint_50_epochs_gans_vt.pt'
    critic_path = ''
    params = [model_path, critic_path]
    fresh = True
    epochs = 1000
    lr = 1E-3
    batch = 1
    device = 'cuda'
    n = 2
    dropout = 0

    # trn(checkpoint_path, epochs=epochs, lr=lr, batch=batch, device=device, n=n,
    #     params=[None, None] if fresh else [f"{checkpoint_path}{model_path}",
    #                                        f"{checkpoint_path}{critic_path}"],
    #     dropout=dropout, HYAK=HYAK, identity=args.identity)

    # Uncomment to validate
    validate(checkpoint_path, model_path=model_path, epochs=2,
             dropout=dropout, batch=batch, n=n, device=device, HYAK=HYAK,
             gui=args.gui)
