import logging
import time
from copy import deepcopy
import os
from functools import wraps

import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

logging.basicConfig(
    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)


def timer(func):
    @wraps(func)
    def timed_func(*args, **kwargs):
        start = time.time_ns()
        val = func(*args, **kwargs)
        dt_ms = (time.time_ns() - start) / 1_000_000
        return dt_ms, val

    return timed_func


class DLDMD(torch.nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        encoding_weight,
        reconstruction_weight,
        prediction_weight,
        phase_space_weight,
        dmd,
        n_prediction_snapshots=1,
        optimizer=optim.Adam,
        optimizer_kwargs={"lr": 1.0e-3, "weight_decay": 1.0e-9},
        epochs=1000,
        print_every=10,
        batch_size=256,
        label="dldmd_run",
        eval_on_cpu=True,
        print_prediction_loss=False,
    ):
        super().__init__()

        self._encoder = encoder
        self._decoder = decoder

        self._encoding_weight = encoding_weight
        self._reconstruction_weight = reconstruction_weight
        self._prediction_weight = prediction_weight
        self._phase_space_weight = phase_space_weight

        if isinstance(epochs, float):
            self._epochs = 100000000
            self._acceptable_loss = epochs
        else:
            self._epochs = epochs
            self._acceptable_loss = 0.0

        self._optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        logging.info(f"Optimizer: {self._optimizer}")

        self._print_every = print_every
        self._batch_size = batch_size
        self._label = label
        self._eval_on_cpu = eval_on_cpu
        self._print_prediction_loss = print_prediction_loss

        logging.info(f"DMD instance: {type(dmd)}")
        self._dmd = dmd
        # a copy of dmd to be used only for evaluation
        self._eval_dmd = deepcopy(dmd)

        self._n_prediction_snapshots = n_prediction_snapshots
        logging.info(
            f"DMD will predict {n_prediction_snapshots} snapshots during training."
        )

        logging.info("----- DLDMD children -----")
        logging.info(tuple(self.children()))

    def forward(self, input):
        if input.ndim == 2:
            input = input[None]

        logging.debug(f"Input shape: {input.shape}")
        encoded_input = self._encoder(input).swapaxes(-1, -2)
        logging.debug(f"Encoded input shape: {encoded_input.shape}")
        self._dmd.fit(encoded_input)
        self._dmd.dmd_time["tend"] = (
            self._dmd.original_time["tend"] + self._n_prediction_snapshots
        )

        encoded_output = self._dmd.reconstructed_data.swapaxes(-1, -2)
        logging.debug(f"Encoded output shape: {encoded_output.shape}")

        if not torch.is_complex(input):
            old_dtype = encoded_output.dtype
            encoded_output = encoded_output.real
            logging.debug(
                f"Removing complex part from output_immersion: {old_dtype} to {encoded_output.dtype}"
            )
        if encoded_output.dtype != input.dtype:
            logging.debug(
                f"Casting output_immersion dtype from {encoded_output.dtype} to {input.dtype}"
            )
            encoded_output = encoded_output.to(dtype=input.dtype)

        return self._decoder(encoded_output)

    def _dmd_training_snapshots(self, snapshots):
        """Batch x Time x Space"""
        if self._n_prediction_snapshots > 0:
            return snapshots[..., : -self._n_prediction_snapshots, :]
        return snapshots

    def _prediction_snapshots(self, snapshots):
        """Batch x Time x Space"""
        if self._n_prediction_snapshots > 0:
            return snapshots[..., -self._n_prediction_snapshots :, :]
        return torch.zeros(0)

    def _compute_loss(self, output, input):
        logging.debug(f"Input shape: {input.shape}")
        logging.debug(f"Output shape: {output.shape}")

        decoder_loss = mse_loss(self._decoder(self._encoder(input)), input)

        batched_psp = self._dmd.operator.phase_space_prediction
        psp_loss = torch.linalg.matrix_norm(batched_psp).sum()

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
    def _train_step(self, loader):
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
    def _eval_step(self, loader):
        self.eval()
        loss_sum = 0.0
        prediction_sum = 0.0
        for i, minibatch in enumerate(loader):
            output = self(self._dmd_training_snapshots(minibatch))
            loss = self._compute_loss(output, minibatch)
            loss_sum += loss.item()

            if self._print_prediction_loss:
                prediction_sum += mse_loss(
                    self._prediction_snapshots(output),
                    self._prediction_snapshots(minibatch),
                ).item()

        loss_avg = loss_sum / (i + 1)
        if self._print_prediction_loss:
            return loss_avg, prediction_sum / (i + 1)
        return loss_avg

    def _save_model(self, label="dldmd"):
        temp = self._dmd
        self._dmd = self._eval_dmd
        torch.save(self, label + ".pl")
        self._dmd = temp

    def _load_model_for_eval(self, label="dldmd"):
        map_location = (
            "cpu"
            if self._eval_on_cpu or not torch.cuda.is_available()
            else "cuda"
        )
        return torch.load(label + ".pl", map_location=map_location)

    def fit(self, X):
        """
        Compute the Deep-Learning enhanced DMD on the input data.

        :param X: the input dataset as a dict with keys `'training_data'` and `test_data`.
        :type X: dict
        """
        train_dataloader = DataLoader(
            X["training_data"], batch_size=self._batch_size, shuffle=True
        )
        test_dataloader = DataLoader(
            X["test_data"], batch_size=self._batch_size, shuffle=True
        )

        train_loss_arr = []
        eval_loss_arr = []

        for epoch in range(1, self._epochs + 1):
            training_time, train_loss = self._train_step(train_dataloader)
            train_loss_arr.append(train_loss)

            model_label = f"{self._label}_e{epoch}"
            self._save_model(model_label)
            eval_model = self._load_model_for_eval(model_label)

            eval = eval_model._eval_step(test_dataloader)
            if self._print_prediction_loss:
                eval_time, (eval_loss, prediction_loss) = eval
            else:
                eval_time, eval_loss = eval

            if not eval_loss_arr or min(eval_loss_arr) > eval_loss:
                self._save_model(f"{self._label}_best")
            eval_loss_arr.append(eval_loss)

            if epoch % self._print_every == 0:
                logging.info(
                    f"[{epoch}] loss: {eval_loss:.4f}, train_time: {training_time:.2f} ms, eval_time: {eval_time:.2f} ms"
                )
                if self._print_prediction_loss:
                    logging.info(
                        f"[{epoch}] prediction loss: {prediction_loss:.4f}"
                    )
            else:
                os.remove(model_label + ".pl")

            if eval_loss < self._acceptable_loss:
                break

        np.save(f"{self._label}_train_loss.npy", train_loss_arr)
        np.save(f"{self._label}_eval_loss.npy", eval_loss_arr)

        return self