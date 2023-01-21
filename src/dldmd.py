import logging
from copy import deepcopy

import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.nn import Module

from utils import timer

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DLDMD(Module):
    def __init__(
        self,
        encoder,
        dmd,
        decoder,
        encoding_weight,
        reconstruction_weight,
        prediction_weight,
        phase_space_weight,
        optimizer=optim.Adam,
        optimizer_kwargs={"lr": 1.0e-3, "weight_decay": 1.0e-9},
        n_prediction_snapshots=1,
        eval_on_cpu=True,
    ):
        super().__init__()

        self._encoder = encoder
        self._decoder = decoder

        self._optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        logger.info(f"Optimizer: {self._optimizer}")

        self._encoding_weight = encoding_weight
        self._reconstruction_weight = reconstruction_weight
        self._prediction_weight = prediction_weight
        self._phase_space_weight = phase_space_weight

        self._eval_on_cpu = eval_on_cpu

        logger.info(f"DMD instance: {type(dmd)}")
        self._dmd = dmd
        # a copy of dmd to be used only for evaluation
        self._eval_dmd = deepcopy(dmd)

        self._n_prediction_snapshots = n_prediction_snapshots
        logger.info(
            f"DMD will predict {n_prediction_snapshots} snapshots during training."
        )

        logger.info("----- DLDMD children -----")
        logger.info(tuple(self.children()))

    def forward(self, input):
        if input.ndim == 2:
            input = input[None]

        logger.debug(f"Input shape: {input.shape}")
        encoded_input = self._encoder(input.swapaxes(-1, -2))
        logger.debug(f"Encoded input shape: {encoded_input.shape}")
        self._dmd.fit(encoded_input.swapaxes(-1, -2), batch=True)
        self._dmd.dmd_time["tend"] = (
            self._dmd.original_time["tend"] + self._n_prediction_snapshots
        )

        encoded_output = self._dmd.reconstructed_data.swapaxes(-1, -2)
        logger.debug(f"Encoded output shape: {encoded_output.shape}")

        if not torch.is_complex(input):
            old_dtype = encoded_output.dtype
            encoded_output = encoded_output.real
            logger.debug(
                f"Removing complex part from output_immersion: {old_dtype} to {encoded_output.dtype}"
            )
        if encoded_output.dtype != input.dtype:
            logger.debug(
                f"Casting output_immersion dtype from {encoded_output.dtype} to {input.dtype}"
            )
            encoded_output = encoded_output.to(dtype=input.dtype)

        return self._decoder(encoded_output).swapaxes(-1, -2)

    def _dmd_training_snapshots(self, snapshots):
        if self._n_prediction_snapshots > 0:
            return snapshots[..., :, : -self._n_prediction_snapshots]
        return snapshots

    def _prediction_snapshots(self, snapshots):
        if self._n_prediction_snapshots > 0:
            return snapshots[..., :, -self._n_prediction_snapshots :]
        return torch.zeros(0)

    def _compute_loss(self, output, input):
        logger.debug(f"Input shape: {input.shape}")
        logger.debug(f"Output shape: {output.shape}")

        decoder_loss = mse_loss(
            self._decoder(self._encoder(input.swapaxes(-1, -2))).swapaxes(
                -1, -2
            ),
            input,
        )

        psp_loss = self._dmd.operator._dmd_phase_space_error.sum()

        reconstruction_loss = mse_loss(
            self._dmd_training_snapshots(output),
            self._dmd_training_snapshots(input),
        )

        prediction_loss = mse_loss(
            self._prediction_snapshots(output),
            self._prediction_snapshots(input),
        )

        return (
            self._encoding_weight * decoder_loss
            + self._phase_space_weight * psp_loss
            + self._reconstruction_weight * reconstruction_loss
            + self._prediction_weight * prediction_loss
        )

    @timer
    def train_step(self, loader):
        self.train()
        loss_sum = 0.0
        for i, minibatch in enumerate(loader):
            self._optimizer.zero_grad()
            output = self(self._dmd_training_snapshots(minibatch))
            loss = self._compute_loss(output, minibatch)
            loss.backward()
            self._optimizer.step()
            loss_sum += loss.item()
        return loss_sum / (i + 1)

    @timer
    def eval_step(self, loader):
        self.eval()
        loss_sum = 0.0
        prediction_sum = 0.0
        for i, minibatch in enumerate(loader):
            output = self(self._dmd_training_snapshots(minibatch))
            loss = self._compute_loss(output, minibatch)
            loss_sum += loss.item()

            prediction_sum += mse_loss(
                self._prediction_snapshots(output),
                self._prediction_snapshots(minibatch),
            ).item()

        loss_avg = loss_sum / (i + 1)
        return loss_avg, prediction_sum / (i + 1)

    def save_model(self, label="dldmd"):
        temp = self._dmd
        self._dmd = self._eval_dmd
        torch.save(self, label + ".pl")
        self._dmd = temp

    @staticmethod
    def load_model_for_eval(label="dldmd", eval_on_cpu=True):
        map_location = (
            "cpu" if eval_on_cpu or not torch.cuda.is_available() else "cuda"
        )
        model = torch.load(label + ".pl", map_location=map_location)
        model.eval()
        return model
