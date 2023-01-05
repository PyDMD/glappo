import argparse
import logging

logging.getLogger().setLevel(logging.INFO)

from torch.nn import Sequential, ReLU, Linear, Module

import torch
from pydmd import DLDMD, DMD

from data import data_maker_fluid_flow_full

# ----------------- CLI --------------------

parser = argparse.ArgumentParser(description="DLDMD command line interface")
parser.add_argument(
    "--hidden",
    help="Encoder hidden layer size",
    type=int,
    default="128",
)
parser.add_argument(
    "--immersion",
    help="Immersion size",
    type=int,
    default="5",
)
parser.add_argument("--svd_rank", type=float, default=-1)
parser.add_argument(
    "--printevery",
    help="Number of epochs to skip before printing evaluation loss",
    type=int,
    default="10",
)
parser.add_argument(
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
    "--training",
    help="Number of samples to be used for training",
    default=10_000,
    type=int,
)
parser.add_argument(
    "--eval",
    help="Number of samples to be used for the evaluation",
    default=3_000,
    type=int,
)
parser.add_argument(
    "--eval_on_cpu",
    help="If set, eval is carried out on CPU",
    action="store_true",
)
parser.add_argument(
    "--print_prediction_loss",
    help="If set, prediction loss is printed in addition to loss",
    action="store_true",
)
parser.add_argument(
    "--stop",
    help="Number of epochs or required accuracy",
    default=1000,
    type=float,
)
parser.add_argument(
    "--label",
    help="Label to be used for serialization",
    required=True,
    type=str,
)
parser.add_argument(
    "--n_prediction_snapshots",
    help="Number of snapshots to be predicted",
    default=1,
    type=int,
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
        eval_on_cpu=args.eval_on_cpu,
        label=args.label,
        print_prediction_loss=args.print_prediction_loss,
        n_prediction_snapshots=args.n_prediction_snapshots
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
data_dict = {
    "training_data": data[: args.training],
    "test_data": data[args.training :],
}

dldmd = allocate_dldmd(data.shape[-1])
if torch.cuda.is_available():
    device = torch.device("cuda")

    data_dict["training_data"] = data_dict["training_data"].to(device)
    if not args.eval_on_cpu:
        data_dict["test_data"] = data_dict["test_data"].to(device)
    else:
        data_dict["test_data"] = data_dict["test_data"].clone()
        del data

    dldmd = dldmd.to(device, dtype=data.dtype)
else:
    dldmd = dldmd.to(dtype=data.dtype)

dldmd.fit(data_dict)