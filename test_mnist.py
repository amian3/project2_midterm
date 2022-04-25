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

        # Initializing all neural network weights.
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
                    
BATCH_SIZE = 32
generator = torch.load("D:/nyu/project3/Generator.pth")
for i in range(10):
  fixed_noise = torch.randn([BATCH_SIZE, 100])
  #fixed_conditional = torch.randint(0, 10, (BATCH_SIZE,))
  fixed_conditional = torch.full((BATCH_SIZE,), i)
  print(fixed_conditional)
  with torch.no_grad():
# Switch model to eval mode.
    generator.eval()
    sr = generator(fixed_noise, fixed_conditional)
    vutils.save_image(sr.detach(), os.path.join("GAN_epoch1.png"), normalize=True)
    lena = mpimg.imread('GAN_epoch1.png')
    plt.imshow(lena) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()