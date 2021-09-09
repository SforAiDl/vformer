import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self, config=(1024,), n_classes=10):
        super(MLPDecoder, self).__init__()

        self.decoder = nn.ModuleList()

        if len(config) > 1:
            for i in range(len(config) - 1):
                self.decoder.append(nn.LayerNorm(config[i]))
                self.decoder.append(nn.Linear(config[i], config[i + 1]))

        self.decoder.append(nn.LayerNorm(config[-1]))
        self.decoder.append(nn.Linear(config[-1], n_classes))

    def forward(self, x):

        return self.decoder(x)
