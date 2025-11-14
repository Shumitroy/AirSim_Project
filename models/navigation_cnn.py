
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NavigationCNN(nn.Module):
    def __init__(self, num_actions: int = 5) -> None:
        """
        Args:
            num_actions: e.g. [forward, left, right, up, down] or any custom set.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # assumes 64x64 input after resizing
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] image tensor (RGB).
        Returns: [B, num_actions] action scores / logits.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    # Quick sanity check
    model = NavigationCNN(num_actions=5)
    dummy_input = torch.randn(2, 3, 64, 64)
    out = model(dummy_input)
    print("Output shape:", out.shape)
