import argparse
import logging

logging.getLogger().setLevel(logging.INFO)

import torch
from pydmd import DMD

from data import data_maker_fluid_flow_full, data_maker_duffing
from dldmd import DLDMD
from mlp import MLP

from train import train_dldmd

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
    required=True
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
    "--epochs",
    help="Number of epochs",
    default=1000,
    type=int,
)
parser.add_argument(
    "--maxloss",
    help="Maximum acceptable loss (training stops if achieved)",
    default=0.0,
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
        n_prediction_snapshots=args.n_prediction_snapshots,
    )


data = data_maker_duffing(
    x_lower1=-1,
    x_upper1=1,
    x_lower2=-1,
    x_upper2=1,
    n_ic=args.training + args.eval,
    dt=0.05,
    tf=200,
).swapaxes(-1,-2)
data = torch.from_numpy(data)
training_data = data[: args.training].clone()
test_data = data[args.training :].clone()
del data

dldmd = allocate_dldmd(training_data.shape[-2])
if torch.cuda.is_available():
    training_data = training_data.cuda()
    if not args.eval_on_cpu:
        test_data = test_data.cuda()

    dldmd = dldmd.to("cuda", dtype=training_data.dtype)
else:
    dldmd = dldmd.to(dtype=training_data.dtype)

train_dldmd(
    dldmd=dldmd,
    training_data=training_data,
    test_data=test_data,
    epochs=args.epochs,
    max_loss=args.maxloss,
    print_prediction_loss=args.print_prediction_loss,
    print_every=args.printevery,
    label=args.label,
    eval_on_cpu=args.eval_on_cpu,
)
