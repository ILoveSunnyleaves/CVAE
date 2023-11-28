import time
import torch

from omegaconf import OmegaConf
from typing import List

from torch import Tensor
from torch import nn
from torch.nn import Linear, Conv2d, ConvTranspose2d
from torch.nn import Sequential, BatchNorm2d, LeakyReLU, Tanh
from torch.nn import functional as F

from utils import get_dataset, save_latest, load_latest
from utils import get_taus, visualize_vae


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


random_seed = 123
torch.manual_seed(random_seed)


"""# CVAE"""

class VAE(nn.Module):
    def __init__(self, config) -> None:
        super(VAE, self).__init__()

        img_size = config.img_size
        num_classes = config.num_classes
        TOPT_CYCLE = config.TOPT_CYCLE

        if TOPT_CYCLE > 0:
            class_dims = 512
            tau_dims = img_size * img_size - class_dims
            self.embed_tau = Linear(TOPT_CYCLE, tau_dims)
        else:
            class_dims = img_size * img_size

        self.embed_class = Linear(num_classes, class_dims)
        self.embed_data = Conv2d(3, 3, kernel_size=1)  # TODO

        hidden_dims = config.model.hidden_dims
        in_channels = 4  # With an extra label channel, i.e. 4=3+1

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            block = Sequential(
                        Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                        BatchNorm2d(h_dim), LeakyReLU()
                    )
            modules.append(block)
            in_channels = h_dim

        self.encoder = Sequential(*modules)

        # Build Decoder
        modules = []
        latent_dim = hidden_dims[-1] * 4  # TODO
        self.latent_dim = latent_dim

        self.decoder_input = Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            block = Sequential(
                ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                kernel_size=3, stride=2, padding=1, output_padding=1),
                BatchNorm2d(hidden_dims[i + 1]), LeakyReLU()
            )
            modules.append(block)

        self.decoder = Sequential(*modules)

        self.final_layer = Sequential(
            ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                            kernel_size=3, stride=2, padding=1, output_padding=1),
            BatchNorm2d(hidden_dims[-1]), LeakyReLU(),
            Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            Tanh()
        )

        # Decoder to predict y.
        self.dy_fc_2 = Linear(latent_dim, num_classes)

        # Decoder to predict tau.
        self.dtau_fc_1 = Linear(latent_dim, latent_dim // 2)
        self.dtau_bn_1 = nn.BatchNorm1d(latent_dim // 2)
        self.dtau_fc_2 = Linear(latent_dim // 2, TOPT_CYCLE)

    def encode(self, input_: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input_: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input_)
        result = torch.flatten(result, start_dim=1)
        return result

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def decode_y(self, z) -> Tensor:
        return self.dy_fc_2(z)

    def decode_tau(self, z) -> Tensor:
        z = self.dtau_fc_1(z)
        z = F.leaky_relu(z, negative_slope=0.2, inplace=True)
        z = self.dtau_bn_1(z)
        tau = self.dtau_fc_2(z)

        return tau

    def forward(self, input_: Tensor, targets: Tensor, taus=None) -> List[Tensor]:
        y = F.one_hot(targets, num_classes=2).float()

        embedded_input = self.embed_data(input_)
        img_size = input_.shape[-1]

        if taus is not None:
            tau = taus.float()
            embedded_class = self.embed_class(y)
            embedded_taus = self.embed_tau(tau)
            embedded_aside = torch.cat([embedded_class, embedded_taus], dim=1)
            embedded_aside = embedded_aside.view(-1, img_size, img_size).unsqueeze(1)
        else:
            embedded_class = self.embed_class(y)
            embedded_class = embedded_class.view(-1, img_size, img_size).unsqueeze(1)
            embedded_aside = embedded_class

        x = torch.cat([embedded_input, embedded_aside], dim=1)
        z = self.encode(x)
        preds_ = self.decode_y(z)
        preds_tau = self.decode_tau(z)
        # z = torch.cat([z, y], dim=1)

        return [self.decode(z), input_, preds_, preds_tau]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, **kwargs)[0]


"""## Inference"""


def inference(model, test_loader, name, epoch, tgt_attr=None, TOPT_CYCLE=None, device=None):
    model.eval()
    torch.manual_seed(random_seed)

    accur = []
    for batch_idx, (features, targets) in enumerate(test_loader):
        features = features.to(device)
        targets = targets[:, tgt_attr].type(torch.int64).to(device)
        _, taus_onehot = get_taus(
            batch_size=features.shape[0], TOPT_CYCLE=TOPT_CYCLE, device=device
        )

        output_, input_, preds_, _ = model(features, targets, taus=taus_onehot)
        _, pred_logits = torch.max(preds_, 1)
        acc = sum(pred_logits.reshape(-1) == targets).item() / features.shape[0]

        accur.append(acc)

    # ACCURACY
    print("Accuracy: ", sum(accur) / len(accur))
    
    visualize_vae(input_, output_, name, epoch)

    model.train()


"""## Training"""

# Commented out IPython magic to ensure Python compatibility.
def training_cae(model, train_ld, test_ld, config, tgt_attr, device):
    name = config.data.name
    n_epochs = config.train.n_epochs
  
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    ce_loss = nn.CrossEntropyLoss()

    # Load previous
    f, epoch_start = load_latest()

    if f is None and epoch_start is None:
        epoch_start = 1
    else:
        model.load_state_dict(torch.load(f))

    ##################

    inference(model, test_ld, name, 0, tgt_attr, config.TOPT_CYCLE, device)

    start_time = time.time()

    for epoch in range(epoch_start, n_epochs + 1):
        for batch_idx, (features, targets) in enumerate(train_ld):
            # don't need labels, only the images (features)
            features = features.to(device)
            targets = targets[:, tgt_attr].type(torch.int64).to(device)
            taus, taus_onehot = get_taus(
                batch_size=features.shape[0], TOPT_CYCLE=config.TOPT_CYCLE, device=device
            )

            output_, input_, preds_, preds_tau = model(
                features, targets=targets, taus=taus_onehot
            )
            cost = F.mse_loss(output_, input_) + 0.0005 * ce_loss(preds_, targets) + 0.0005 * ce_loss(preds_tau, taus)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if not batch_idx % 100:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                      % (epoch, n_epochs, batch_idx, len(train_ld), cost))

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
        scheduler.step()

        if epoch % 5 == 0:
            inference(model, test_ld, name, epoch, tgt_attr, config.TOPT_CYCLE, device)

        # Save model
        save_latest(model, name, tgt_attr, epoch)

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load("vae_config.yaml")

    data_name = config.data.name

    if data_name == "cifar10":
        tgt_attr = 5  # Cifar10
    else:
        raise NotImplementedError
    
    _, _, train_ld, test_ld = get_dataset(config)
    model = VAE(config).to(device)

    training_cae(model, train_ld, test_ld, config, tgt_attr=tgt_attr, device=device)


if __name__ == '__main__':
    main()
