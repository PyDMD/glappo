import argparse
import logging

logging.getLogger().setLevel(logging.INFO)

from torch.nn import Sequential, ReLU, Linear, Module

import torch
from pydmd import DLDMD, DMD

from data import data_maker_fluid_flow_full

# ----------------- CLI --------------------

parser = argparse.ArgumentParser(help="DLDMD command line interface")
parser.add_argument(
    "-h",
    "--hidden",
    help="Encoder hidden layer size",
    type=int,
    default="128",
)
parser.add_argument(
    "-i",
    "--immersion",
    help="Immersion size",
    type=int,
    default="5",
)
parser.add_argument("-s", "--svd_rank", type=float, default=-1)
parser.add_argument(
    "-p",
    "--printevery",
    help="Number of epochs to skip before printing evaluation loss",
    type=int,
    default="10",
)
parser.add_argument(
    "-a",
    "--anomaly",
    help="Whether to enable or not PyTorch anomaly detection",
    type=bool,
    default=False,
)
parser.add_argument("--encoding_weight", type=float, default=1.0)
parser.add_argument("--reconstruction_weight", type=float, default=1.0)
parser.add_argument("--prediction_weight", type=float, default=1.0)
parser.add_argument("--phase_space_weight", type=float, default=1.0)
parser.add_argument(
    "-t",
    "--training",
    help="Number of samples to be used for training",
    default=10_000,
    type=int,
)
parser.add_argument(
    "-e",
    "--eval",
    help="Number of samples to be used for the evaluation",
    default=3_000,
    type=int,
)
parser.add_argument(
    "--stop",
    help="Number of epochs or required accuracy",
    default=1000,
    type=float,
)
args = parser.parse_args()

# ----------------- CLI --------------------

if args.anomaly:
    torch.autograd.set_detect_anomaly(True)


def save(encoder, decoder, label="0"):
    torch.save(encoder, f"encoder_{label}.pt")
    torch.save(decoder, f"decoder_{label}.pt")


def load(label="0"):
    return torch.load(f"encoder_{label}.pt"), torch.load(f"decoder_{label}.pt")


def allocate_dldmd(input_size):
    encoder = MLP(input_size, args.immersion, args.hidden)
    decoder = MLP(args.immersion, input_size, args.hidden)
    dmd = DMD(svd_rank=args.svd_rank)
    return DLDMD(
        encoder=encoder,
        decoder=decoder,
        encoding_weight=args.encoding_weight,
        reconstruction_weight=args.reconstruction_weight,
        prediction_weight=args.prediction_weight,
        phase_space_weight=args.phase_space_weight,
        dmd=dmd,
        print_every=args.printevery,
        epochs=args.stop,
    )


class MLP(Module):
    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()
        self.layers = Sequential(
            Linear(input_size, hidden_layer_size),
            ReLU(),
            Linear(hidden_layer_size, hidden_layer_size),
            ReLU(),
            Linear(hidden_layer_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


data = data_maker_fluid_flow_full(
    x1_lower=-1.1,
    x1_upper=1.1,
    x2_lower=-1.1,
    x2_upper=1.1,
    x3_lower=0.0,
    x3_upper=2,
    n_ic=args.training + args.eval,
    dt=0.01,
    tf=6,
)
data = torch.from_numpy(data)

dldmd = allocate_dldmd(data.shape[-1])
if torch.cuda.is_available():
    device = torch.device("cuda")
    data = data.to(device)
    dldmd = dldmd.to(device, dtype=data.dtype)
dldmd.fit(
    {"training_data": data[: args.training], "test_data": data[args.training :]}
)

save(dldmd._encoder, dldmd._decoder)
