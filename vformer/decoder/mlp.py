import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    Parameters
    -----------
    config : int or tuple or list
        Dimension of Hidden layer(s)
    n_classes : int
        Number of classes for classification
    """

    def __init__(self, config=(1024,), n_classes=10):
        super(MLPDecoder, self).__init__()

        self.decoder = nn.ModuleList()

        if not isinstance(config, list) and not isinstance(config, tuple):
            config = [config]

        if len(config) > 1:
            for i in range(len(config) - 1):

                self.decoder.append(nn.LayerNorm(config[i]))
                self.decoder.append(nn.Linear(config[i], config[i + 1]))

        self.decoder.append(nn.LayerNorm(config[-1]))
        self.decoder.append(nn.Linear(config[-1], n_classes))

        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):

        return self.decoder(x)
