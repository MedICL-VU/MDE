import torch
import torch.nn as nn


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128]):
        """
        MLP-based GAN discriminator for DINOv2-style features.

        Args:
            input_dim (int): Feature vector size (e.g., 768 from DINOv2)
            hidden_dims (list): Hidden layer dimensions
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            prev_dim = h
        layers += [nn.Linear(prev_dim, 1)]  # Single logit for BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # Output shape: [B, 1]


if __name__ == '__main__':
    D = MLPDiscriminator()
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Fake and real features (from render and real domains)
    real_input = torch.randn(32, 768)
    fake_input = torch.randn(32, 768)

    # Discriminator outputs
    pred_real = D(real_input)  # [32, 1]
    pred_fake = D(fake_input)  # [32, 1]

    # Labels: 1 for real, 0 for fake
    loss_real = adversarial_loss(pred_real, torch.ones_like(pred_real))
    loss_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
    loss_D = 0.5 * (loss_real + loss_fake)

    # Generator loss (wants D(fake) → 1)
    loss_G = adversarial_loss(pred_fake, torch.ones_like(pred_fake))

    print(f"Discriminator loss: {loss_D.item():.4f}")
    print(f"Generator loss: {loss_G.item():.4f}")
