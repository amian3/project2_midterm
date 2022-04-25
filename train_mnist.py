import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import numpy as np
import os
image_size = 28
LR = 0.0001
BATCH_SIZE = 32

class Generator(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10) -> None:
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(100 + num_classes, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(1024, channels * image_size * image_size),
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, labels: list = None) -> torch.Tensor:

        conditional_inputs = torch.cat([inputs, self.label_embedding(labels)], dim=-1)
        out = self.main(conditional_inputs)
        out = out.reshape(out.size(0), self.channels, self.image_size, self.image_size)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class DiscriminatorForMNIST(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10) -> None:
        super(DiscriminatorForMNIST, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(channels * image_size * image_size + num_classes, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, labels: list = None) -> torch.Tensor:
        inputs = torch.flatten(inputs, 1)
        conditional = self.label_embedding(labels)
        conditional_inputs = torch.cat([inputs, conditional], dim=-1)
        out = self.main(conditional_inputs)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

dataset = torchvision.datasets.MNIST(root='./data/', download=True,
      transform=transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
      ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
shuffle= True)

generator = Generator()
discriminator = DiscriminatorForMNIST()
adversarial_criterion = nn.BCELoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))




for epoch in range(50):
  print("epoch:", epoch)
  discriminator.train()
  generator.train()
  for i, (inputs, target) in enumerate(dataloader):



    real_label = torch.full((BATCH_SIZE, 1), 1, dtype=inputs.dtype)
    fake_label = torch.full((BATCH_SIZE, 1), 0, dtype=inputs.dtype)

    noise = torch.randn([BATCH_SIZE, 100])
    conditional = torch.randint(0, 10, (BATCH_SIZE,))

    
    
    
    discriminator.zero_grad()
    # Train with real.
    real_output = discriminator(inputs, target)
    d_loss_real = adversarial_criterion(real_output, real_label)
    d_loss_real.backward()
    d_x = real_output.mean()


    # Train with fake.
    fake = generator(noise, conditional)
    fake_output = discriminator(fake.detach(), conditional)
    d_loss_fake = adversarial_criterion(fake_output, fake_label)
    d_loss_fake.backward()
    d_g_z1 = fake_output.mean()

    d_loss = d_loss_real + d_loss_fake
    discriminator_optimizer.step()
    
    generator.zero_grad()

    fake_output = discriminator(fake, conditional)
    g_loss = adversarial_criterion(fake_output, real_label)
    g_loss.backward()
    d_g_z2 = fake_output.mean()
    generator_optimizer.step()

torch.save(generator.state_dict(), os.path.join("Generator.pth"))
torch.save(discriminator.state_dict(), os.path.join("Discriminator.pth"))