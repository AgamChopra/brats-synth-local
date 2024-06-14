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
from utils import grad_penalty

# Set PyTorch printing precision
torch.set_printoptions(precision=8)

# Set precision for matrix multiplication
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train(checkpoint_path, epochs=200, lr=1E-4, batch=1,
          device='cpu', model_path=None, critic_path=None, n=1,
          beta1=0.5, beta2=0.999, Lambda_penalty=10, dropout=0.1, HYAK=False):
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
    neural_network = models.Attention_UNetT(
        in_c=2, out_c=1, n=n, dropout_rate=dropout, vision_transformer=True).to(device)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        neural_network.load_state_dict(state_dict, strict=True)

    critic = models.Critic_VT(in_c=3, fact=1).to(device)
    if critic_path is not None:
        try:
            state_dict = torch.load(critic_path, map_location=device)
            critic.load_state_dict(state_dict, strict=True)
        except Exception:
            print(f"{critic_path} not found!")

    # Initialize optimizers
    optimizer = torch.optim.AdamW(neural_network.parameters(), lr)
    optimizerC = torch.optim.AdamW(
        critic.parameters(), lr, betas=(beta1, beta2))

    # Initialize loss functions and criteria
    criterion_masked = [Mask_MSELoss(), Mask_L1Loss()]
    criterion = [ssim_loss(win_size=3, win_sigma=0.1),
                 GMELoss3D(device=device)]
    lambdas = [0.5, 0.5]

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
    critic_losses, critic_losses_train = [], []
    mse, mae, psnr, ssim = [], [], [], []
    mse_val, mae_val, psnr_val, ssim_val = [], [], [], []

    scaler = GradScaler()

    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch}:')
        neural_network.train()
        critic.train()

        for _ in trange(iterations):
            x, mask = data.load_batch()
            if x is None or mask is None:
                print("Batch returned None values.")
                continue

            x, mask = x.float().to(device), mask.float().to(device)
            whole_mask = (x > 0.).float().to(device)

            known_masked_region = x * mask
            input_image = (x > 0) * ((mask < 0.5) * x)
            input_image = torch.cat((input_image, mask), dim=1)

            if (epoch + 1) % 2 == 0:
                optimizer.zero_grad()
                with autocast():
                    y = whole_mask * \
                        neural_network(input_image, gans=True).float()
                    synthetic_masked_region = mask * y

                    error = sum([efunc(known_masked_region, synthetic_masked_region, mask)
                                for efunc in criterion_masked])
                    error += sum([lambdas[i] * criterion[i](x, y)
                                 for i in range(len(criterion))])
                    error -= critic(torch.cat((input_image,
                                    synthetic_masked_region), dim=1).float()).mean()

                scaler.scale(error).backward()
                scaler.step(optimizer)
                scaler.update()

                losses.append(error.item())

            else:
                optimizerC.zero_grad()
                with autocast():
                    y = whole_mask * \
                        neural_network(input_image, gans=True).float()
                    synthetic_masked_region = mask * y

                    real_x = torch.cat(
                        (input_image, known_masked_region), dim=1).float()
                    fake_x = torch.cat(
                        (input_image, synthetic_masked_region), dim=1).float()

                    error_fake = critic(fake_x.detach()).mean()
                    error_real = critic(real_x).mean()
                    penalty = grad_penalty(
                        critic, real_x, fake_x, Lambda_penalty)

                    error = error_fake - error_real + penalty

                scaler.scale(error).backward(retain_graph=True)
                scaler.step(optimizerC)
                scaler.update()

                critic_losses.append(error.item())

        losses_train.append(sum(losses) / len(losses) if losses else 0)
        losses = []

        critic_losses_train.append(
            sum(critic_losses) / len(critic_losses) if critic_losses else 0)
        critic_losses = []

        # Validation loop
        neural_network.eval()
        for _ in trange(3 * iterations_val):
            with torch.no_grad():
                x, mask = data_val.load_batch()
                x, mask = x.float().to(device), mask.float().to(device)
                whole_mask = (x > 0.).float().to(device)

                known_masked_region = x * mask
                input_image = (x > 0) * ((mask < 0.5) * x)
                input_image = torch.cat((input_image, mask), dim=1)

                with autocast():
                    y = whole_mask * neural_network(input_image).float()
                    synthetic_masked_region = mask * y

                    error = 10 * \
                        sum([efunc(known_masked_region, synthetic_masked_region, mask)
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
            train_visualize([critic_losses_train, losses_train, losses_val, mae_val,
                            mse_val, ssim_val, psnr_val], gans=True, dpi=180, HYAK=HYAK)

        if epoch % 10 == 0 and epoch != 0:
            torch.save(neural_network.state_dict(),
                       f"{checkpoint_path}checkpoint_{epoch}_epochs.pt")
            torch.save(critic.state_dict(),
                       f"{checkpoint_path}critic_checkpoint_{epoch}_epochs.pt")

        if mse_val[-1] >= max(mse_val):
            torch.save(neural_network.state_dict(),
                       f"{checkpoint_path}best_mse.pt")
            best_mse = epoch

        if ssim_val[-1] >= max(ssim_val):
            torch.save(neural_network.state_dict(),
                       f"{checkpoint_path}best_ssim.pt")
            best_ssim = epoch

        if psnr_val[-1] >= max(psnr_val):
            torch.save(neural_network.state_dict(),
                       f"{checkpoint_path}best_psnr.pt")
            best_psnr = epoch

        norm_metrics = array([norm(mse_val), norm(ssim_val), norm(psnr_val)])
        avg_metrics = norm_metrics.sum(axis=0)

        if avg_metrics[-1] >= avg_metrics.max():
            torch.save(neural_network.state_dict(),
                       f"{checkpoint_path}best_average.pt")
            best_avg = epoch

        print(
            f'Average Train Loss: {losses_train[-1]:.6f}, Validation Loss: {losses_val[-1]:.6f}')
        print(
            f'Validation MSE: {mse_val[-1]:.6f}, MAE: {mae_val[-1]:.6f}, PSNR: {psnr_val[-1]:.6f}, SSIM: {ssim_val[-1]:.6f}')
        print(
            f'Best epochs for mse: {best_mse}, ssim: {best_ssim}, psnr: {best_psnr}, norm_average: {best_avg}')

    torch.save(neural_network.state_dict(),
               f"{checkpoint_path}checkpoint_{epochs}_epochs.pt")

    return losses_train, losses_val


def validate(checkpoint_path, model_path, batch=1, n=1, epochs=100, dropout=0.1, device='cpu', HYAK=False):
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
    print(device)

    model = models.Attention_UNetT(
        in_c=2, out_c=1, n=n, dropout_rate=dropout, vision_transformer=True).to(device)
    state_dict = torch.load(
        f"{checkpoint_path}{model_path}", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

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
                x, mask = x.to(device).float(), mask.to(device).float()
                whole_mask = (x > 0.).float().to(device)

                with autocast():
                    input_image = (x > 0) * ((mask < 0.5) * x)
                    input_image = torch.cat((input_image, mask), dim=1)
                    y = torch.clip(model(input_image, gans=True), 0., 1.)
                    synthetic_masked_region = mask * y * whole_mask

                    score = [scores[i](
                        x, x * (mask < 0.5) + synthetic_masked_region) for i in range(len(scores))]

                mse.append(score[0].item())
                mae.append(score[1].item())
                psnr.append(score[2].item())
                ssim.append(score[3].item())

                dataloader.show_images(
                    torch.cat(
                        (x.cpu(), input_image[:, 0:1].cpu(), (synthetic_masked_region + (x * (mask < 0.5))).cpu(),
                         y.cpu(), synthetic_masked_region.cpu(), torch.abs(x * mask - synthetic_masked_region).cpu()), dim=0), 6, 3, dpi=350)

        print(f'\n{model_path}')
        print(
            f'Total Average MSE: {sum(mse) / len(mse):.4f}, MAE: {sum(mae) / len(mae):.4f}, PSNR: {sum(psnr) / len(psnr):.4f}, SSIM: {sum(ssim) / len(ssim):.4f}')


def trn(checkpoint_path, epochs=500, lr=1E-4, batch=1, device='cpu', n=1, params=None, dropout=0.1, HYAK=False):
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
    HYAK = True
    checkpoint_path = '/gscratch/kurtlab/brats2024/repos/agam/brats-synth-local/log/' if HYAK else '/home/agam/Documents/git-files/brats-synth-local/'
    model_path = 'best_average.pt'
    critic_path = 'critic.pt'
    params = [model_path, critic_path]
    fresh = True
    epochs = 1000
    lr = 1E-3
    batch = 1
    device = 'cuda'
    n = 1
    dropout = 0

    trn(checkpoint_path, epochs=epochs, lr=lr, batch=batch, device=device, n=n,
        params=[None, None] if fresh else [f"{checkpoint_path}{model_path}", f"{checkpoint_path}{critic_path}"], dropout=dropout, HYAK=HYAK)

    # Uncomment to validate
    # validate(checkpoint_path, model_path=model_path, epochs=2, dropout=dropout, batch=batch, n=n, device=device, HYAK=HYAK)
