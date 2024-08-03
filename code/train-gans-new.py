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
# from torch.cuda.amp import GradScaler, autocast
from numpy import array
from tqdm import trange

import dataloader
import models
from grokfast import gradfilter_ema
from utils import PSNR_Metric, SSIM_Metric, train_visualize, norm
from utils import SSIMLoss, GMELoss3D, grad_penalty, show_images

# Set PyTorch printing precision
torch.set_printoptions(precision=8)

# Set precision for matrix multiplication
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train(checkpoint_path, epochs=200, lr=1E-4, batch=1,
          model_path=None, critic_path=None,
          beta1=0.5, beta2=0.999, Lambda_penalty=10, dropout=0.1,
          HYAK=False, identity='', device1='cpu', device2='cpu',
          agrok_lpha=0.98, grok_lamb=2., accumulated_batch=32):
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
        beta1 (float, optional): Beta1 for Adam optimizer. Default is 0.5.
        beta2 (float, optional): Beta2 for Adam optimizer. Default is 0.999.
        Lambda_penalty (int, optional): Lambda for gradient penalty. Default is 10.
        dropout (float, optional): Dropout rate. Default is 0.1.
        HYAK (bool, optional): Flag to use HYAK paths. Default is False.
    """

    # Initialize models
    generator = models.Global_UNet(
        in_c=1,
        out_c=1,
        fact=32,
        embed_dim=384,
        n_heads=16,
        mlp_ratio=32,
        qkv_bias=True,
        dropout_rate=0.,
        mask_downsample=16,
        noise=True,
        device1=device1,
        device2=device2
    )
    print(f'Gen. size: {models.count_parameters(generator)/1000000000}B')
    if model_path is not None:
        state_dict = torch.load(model_path)
        generator.load_state_dict(state_dict, strict=True)

    critic = models.CriticA(in_c=2, fact=2).to(device2)
    print(f'Crit. size: {models.count_parameters(critic)/1000000}M')
    if critic_path is not None:
        try:
            state_dict = torch.load(critic_path)
            critic.load_state_dict(state_dict, strict=True)
        except Exception:
            print(f"{critic_path} not found!")

    # Initialize optimizers
    grads = None
    gradsC = None
    optimizer = torch.optim.AdamW(generator.parameters(), lr)
    optimizerC = torch.optim.AdamW(
        critic.parameters(), lr, betas=(beta1, beta2))

    # Initialize loss functions and criteria
    criterion = [SSIMLoss(win_size=3, win_sigma=0.1),
                 GMELoss3D(),
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
    critic_losses_real, critic_losses_fake = [], []
    critic_losses_train_real, critic_losses_train_fake = [], []
    mse, mae, psnr, ssim = [], [], [], []
    mse_val, mae_val, psnr_val, ssim_val = [], [], [], []

    num_batches = int(iterations/accumulated_batch)

    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch}:')
        generator.train()
        critic.train()

        for itervar in range(num_batches):
            error_accum = []
            error_accum_real = []
            error_accum_fake = []
            print(f'Batch {itervar + 1}/{num_batches + 1}:')
            if (itervar + 1) % 5 == 0:
                print('Generator Optimization')
                optimizer.zero_grad()
                for _ in trange(accumulated_batch):
                    x, mask = data.load_batch()
                    if x is None or mask is None:
                        print("Batch returned None values.")
                        continue
                    x, mask = x.float().to(device1), mask.float().to(device1)
                    input_image = (x > 0) * ((mask < 0.5) * x)

                    y = generator(input_image, mask).float()
                    # print(y.mean().item())

                    error_sup = sum([lambdas[i] * criterion[i](x.to(device2), y)
                                     for i in range(len(criterion))])
                    # print(error_sup.item())
                    error_gans = critic(
                        torch.cat((input_image.to(device2), y), dim=1).float()).mean()
                    # print(error_gans.item())

                    error = 10 * error_sup - 0.5 * error_gans
                    error.backward()
                    error_accum.append(error_gans.item())

                grads = gradfilter_ema(
                    generator, grads=grads, alpha=agrok_lpha, lamb=grok_lamb)
                optimizer.step()

                losses.append(sum(error_accum)/len(error_accum))

                print(f'  Error: {losses[-1]}')

            else:
                print('Critic Optimization')
                optimizerC.zero_grad()
                for _ in trange(accumulated_batch):
                    x, mask = data.load_batch()
                    if x is None or mask is None:
                        print("Batch returned None values.")
                        continue
                    x, mask = x.float().to(device1), mask.float().to(device1)
                    input_image = (x > 0) * ((mask < 0.5) * x)

                    y = generator(input_image, mask).float()
                    # print(y.mean().item())

                    real_x = torch.cat(
                        (input_image, x), dim=1).float().to(device2)
                    fake_x = torch.cat(
                        (input_image.to(device2), y.detach()), dim=1).float()

                    error_fake = critic(fake_x).mean()
                    error_real = critic(real_x).mean()
                    penalty = grad_penalty(
                        critic, real_x, fake_x, Lambda_penalty)
                    # print(error_fake.item(),
                    #       error_real.item(),
                    #       penalty.item())

                    error = error_fake - error_real + penalty
                    error.backward(retain_graph=True)
                    error_accum.append(error.item())
                    error_accum_real.append(error_real.item())
                    error_accum_fake.append(error_fake.item())

                gradsC = gradfilter_ema(
                    critic, grads=gradsC, alpha=agrok_lpha, lamb=grok_lamb)
                optimizerC.step()

                critic_losses.append(sum(error_accum)/len(error_accum))
                critic_losses_real.append(
                    sum(error_accum_real)/len(error_accum_real))
                critic_losses_fake.append(
                    sum(error_accum_fake)/len(error_accum_fake))

                print(f'  Error: {critic_losses[-1]}, real:{critic_losses_real[-1]}, fake:{critic_losses_fake[-1]}')

        losses_train.append(sum(losses) / len(losses))
        losses = []

        critic_losses_train.append(sum(critic_losses) / len(critic_losses))
        critic_losses = []

        critic_losses_train_real.append(
            sum(critic_losses_real) / len(critic_losses_real))
        critic_losses_real = []

        critic_losses_train_fake.append(
            sum(critic_losses_fake) / len(critic_losses_fake))
        critic_losses_fake = []

        print(f'Average Train Loss: gen:{losses_train[-1]:.6f} crit:{critic_losses_train[-1]:.6f}, real:{critic_losses_train_real[-1]:.6f}, fake:{critic_losses_train_fake[-1]:.6f}')

        # Validation loop
        generator.eval()
        for _ in trange(iterations_val):
            with torch.no_grad():
                x, mask = data.load_batch()
                if x is None or mask is None:
                    print("Batch returned None values.")
                    continue
                x, mask = x.float().to(device1), mask.float().to(device1)
                input_image = (x > 0) * ((mask < 0.5) * x)

                y = generator(input_image, mask).float()

                error_sup = sum([lambdas[i] * criterion[i](x.to(device2), y)
                                 for i in range(len(criterion))])
                error_gans = critic(
                    torch.cat((input_image.to(device2), y), dim=1).float()).mean()

                error = 1E4 * error_sup - 0.1 * error_gans

                losses_temp.append(error.item())
                mse.append(-torch.log10(mse_metric(x.to(device2), y) + 1e-12).item())
                mae.append(-torch.log10(mae_metric(x.to(device2), y) + 1e-12).item())
                ssim.append(-torch.log10(1 -
                            ssim_metric(x.to(device2), y) + 1e-12).item())
                psnr.append(psnr_metric(x.to(device2), y).item())

        show_images(torch.cat([
            x.detach().cpu(),
            input_image.detach().cpu(),
            y.detach().cpu(),
            torch.abs(x-y).detach().cpu()
        ], dim=0), 4, 2, dpi=250)

        mse_val.append(sum(mse) / len(mse))
        mae_val.append(sum(mae) / len(mae))
        ssim_val.append(sum(ssim) / len(ssim))
        psnr_val.append(sum(psnr) / len(psnr))
        losses_val.append(
            sum(losses_temp) / len(losses_temp))
        losses_temp = []
        mse, mae, psnr, ssim = [], [], [], []

        train_visualize([critic_losses_train, losses_train,
                         [critic_losses_train_real,
                             critic_losses_train_fake], mae_val,
                         mse_val, ssim_val, psnr_val],
                        gans=True, dpi=180,
                        path=checkpoint_path if HYAK else None,
                        identity=identity)

        if epoch % 1 == 0:
            torch.save(generator.state_dict(),
                       f"{checkpoint_path}checkpoint_latest_{identity}.pt")
            torch.save(critic.state_dict(),
                       f"{checkpoint_path}critic_checkpoint_latest_{identity}.pt")

        if mse_val[-1] >= max(mse_val) or epoch == 0:
            torch.save(generator.state_dict(),
                       f"{checkpoint_path}best_mse_{identity}.pt")
            best_mse = epoch

        if ssim_val[-1] >= max(ssim_val) or epoch == 0:
            torch.save(generator.state_dict(),
                       f"{checkpoint_path}best_ssim_{identity}.pt")
            best_ssim = epoch

        if psnr_val[-1] >= max(psnr_val) or epoch == 0:
            torch.save(generator.state_dict(),
                       f"{checkpoint_path}best_psnr_{identity}.pt")
            best_psnr = epoch

        norm_metrics = array([norm(mse_val), norm(ssim_val), norm(psnr_val)])
        avg_metrics = norm_metrics.sum(axis=0)

        if avg_metrics[-1] >= avg_metrics.max() or epoch == 0:
            torch.save(generator.state_dict(),
                       f"{checkpoint_path}best_average_{identity}.pt")
            best_avg = epoch

        print(f'Average Validation Loss: {losses_val[-1]:.6f}')
        print(
            f'Validation MSE: {mse_val[-1]:.6f}, MAE: {mae_val[-1]:.6f}, PSNR: {psnr_val[-1]:.6f}, SSIM: {ssim_val[-1]:.6f}')
        print(
            f'Best epochs for mse: {best_mse}, ssim: {best_ssim}, psnr: {best_psnr}, norm_average: {best_avg}')

    torch.save(generator.state_dict(),
               f"{checkpoint_path}checkpoint_final_{epochs}_epochs_{identity}.pt")


def trn(checkpoint_path,
        epochs=200,
        lr=1E-4,
        batch=1,
        accumulated_batch=32,
        params=[None, None],
        dropout=0.1,
        HYAK=False,
        identity='',
        device1='cpu',
        device2='cpu'):
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
    train(
        checkpoint_path=checkpoint_path,
        epochs=epochs,
        lr=lr,
        batch=batch,
        accumulated_batch=accumulated_batch,
        device1=device1,
        device2=device2,
        model_path=params[0],
        critic_path=params[1],
        dropout=dropout,
        HYAK=HYAK,
        identity=identity
    )


def validate(checkpoint_path, model_path, batch=1, epochs=100,
             dropout=0.1, device='cpu', HYAK=False, gui=True):
    """
    Validation function for the model.

    Args:
        checkpoint_path (str): Path to save checkpoints.
        model_path (str): Path to the model checkpoint.
        batch (int, optional): Batch size. Default is 1.
        epochs (int, optional): Number of epochs. Default is 100.
        dropout (float, optional): Dropout rate. Default is 0.1.
        device (str, optional): Device to run the validation on. Default is 'cpu'.
        HYAK (bool, optional): Flag to use HYAK paths. Default is False.
    """
    with torch.no_grad():
        model = models.Global_UNet(
            in_c=1,
            out_c=1,
            fact=64,
            embed_dim=1024,
            n_heads=32,
            mlp_ratio=64,
            qkv_bias=True,
            dropout_rate=0.,
            mask_downsample=30,
            noise=True,
            device1=device,
            device2=device
        )
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

                input_image = (x > 0) * ((mask < 0.5) * x)
                y = model(input_image, mask)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GANs.')
    parser.add_argument('--gui', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='Enable or disable GUI (default: True)')
    parser.add_argument('--hyak', type=lambda x: (str(x).lower() == 'true'),
                        default=False, help='Enable or disable HYAK mode (default: False)')
    parser.add_argument('--identity', default='gans',
                        help='identity of the run (default: gans)')
    parser.add_argument('--device1', default='cuda:0',
                        help='device1 (default: cuda:0)')
    parser.add_argument('--device2', default='cuda:0',
                        help='device2 (default: cuda:0)')
    parser.add_argument('--mode', default='validate',
                        help='mode (train/validate/evaluate) (default: validate)')
    args = parser.parse_args()
    print(f"GUI Enabled: {args.gui}")
    print(f"HYAK Enabled: {args.hyak}")
    print(f"Identity: {args.identity}")
    print(f"Device[1,2]: [{args.device1},{args.device2}]")
    print(f"Mode: {args.mode}")

    HYAK = args.hyak
    checkpoint_path = '/gscratch/kurtlab/brats2024/repos/agam/brats-synth-local/log/' if HYAK else '/home/agam/Desktop/hyak-current-log/'
    model_path = 'checkpoint_latest_{args.identity}.pt'
    critic_path = 'critic_checkpoint_latest_{args.identity}.pt'
    params = [model_path, critic_path]
    fresh = True
    epochs = 1000
    lr = 1E-4
    batch = 1
    accumulated_batch = 32
    device1 = args.device1
    device2 = args.device2
    dropout = 0

    if args.mode == 'train':
        trn(checkpoint_path, epochs=epochs, lr=lr, batch=batch,
            device1=device1, device2=device2, dropout=dropout,
            HYAK=HYAK, identity=args.identity,
            accumulated_batch=accumulated_batch,
            params=[None, None] if fresh else [
                f"{checkpoint_path}{model_path}",
                f"{checkpoint_path}{critic_path}"
            ])

    elif args.mode == 'validate':
        validate(checkpoint_path, model_path=model_path, epochs=1,
                 dropout=dropout, batch=batch, device=device1, HYAK=HYAK,
                 gui=args.gui)
