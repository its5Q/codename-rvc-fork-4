import torch
from torch.nn import functional as F

def phase_loss(x_fft: torch.Tensor, g_fft: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    x_norm = x_fft / (x_fft.abs() + 1e-9)
    g_norm = g_fft / (g_fft.abs() + 1e-9)

    phase_similarity = (x_norm * g_norm.conj()).real
    loss = 1.0 - phase_similarity

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")


def feature_loss(fmap_r, fmap_g):
    """
    Compute the feature loss between reference and generated feature maps.

    Args:
        fmap_r (list of torch.Tensor): List of reference feature maps.
        fmap_g (list of torch.Tensor): List of generated feature maps.
    """
    return 2 * sum(
        torch.mean(torch.abs(rl - gl))
        for dr, dg in zip(fmap_r, fmap_g)
        for rl, gl in zip(dr, dg)
    )


def feature_loss_mask(fmap_r, fmap_g, silence_mask=None, reduce=True):
    """
    Silence-aware feature matching loss.
    If silence_mask is provided, applies it per sample to reduce loss contribution.

    Args:
        fmap_r (List[List[Tensor]]): Feature maps from real audio
        fmap_g (List[List[Tensor]]): Feature maps from generated audio
        silence_mask (Tensor or None): Tensor of shape [B], 1 for voiced, 0 for silence
        reduce (bool): Whether to return mean or per-sample loss
    Returns:
        Scalar loss or per-sample tensor
    """
    losses = []

    for dr, dg in zip(fmap_r, fmap_g):  # across discriminators
        for rl, gl in zip(dr, dg):      # across layers
            diff = torch.abs(rl - gl)
            per_sample = diff.view(diff.shape[0], -1).mean(dim=1)  # [B]
            losses.append(per_sample)

    total = torch.stack(losses, dim=0).mean(dim=0)  # mean over layers → [B]

    if silence_mask is not None:
        total = total * silence_mask  # scale loss per sample

    if reduce:
        return total.sum() / (silence_mask.sum() + 1e-6 if silence_mask is not None else total.numel())
    else:
        return total  # shape [B]


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Compute the discriminator loss for real and generated outputs.

    Args:
        disc_real_outputs (list of torch.Tensor): List of discriminator outputs for real samples.
        disc_generated_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
    """
    loss = 0
    # r_losses = []
    # g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)

        # r_losses.append(r_loss.item())
        # g_losses.append(g_loss.item())
        loss += r_loss + g_loss

    return loss # , r_losses, g_losses


def generator_loss(disc_outputs):
    """
    LSGAN Generator Loss:
    """
    loss = 0
    #gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        # gen_losses.append(l.item())
        loss += l

    return loss #, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Compute the Kullback-Leibler divergence loss.

    Args:
        z_p (torch.Tensor): Latent variable z_p [b, h, t_t].
        logs_q (torch.Tensor): Log variance of q [b, h, t_t].
        m_p (torch.Tensor): Mean of p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()

    return loss
