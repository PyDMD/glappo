import logging
import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

logging.basicConfig(
    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)


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
        prediction_snapshots=1,
        optimizer=optim.Adam,
        optimizer_kwargs={"lr": 1.0e-3, "weight_decay": 1.0e-9},
        epochs=1000,
        print_every=10,
        batch_size=256,
        label="dldmd_run",
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

        logging.info(f"DMD instance: {type(dmd)}")
        self._dmd = dmd

        self._prediction_snapshots = prediction_snapshots
        logging.info(
            f"DMD will predict {prediction_snapshots} snapshots during training."
        )

        logging.info("----- DLDMD children -----")
        logging.info(tuple(self.children()))

    def forward(self, input):
        if input.ndim == 2:
            input = input[None]

        encoded_input = self._encoder(input)
        self._dmd.fit(encoded_input)
        self._dmd.dmd_time["tend"] = (
            self._dmd.original_time["tend"] + self._prediction_snapshots
        )
        encoded_output = self._dmd.reconstructed_data

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

    def _compute_loss(self, output, input):
        decoder_loss = mse_loss(self._decoder(self._encoder(input)), input)

        batched_psp = self._dmd.operator.phase_space_prediction
        psp_loss = torch.linalg.matrix_norm(batched_psp).sum()

        if self._prediction_snapshots > 0:
            reconstruction_loss = mse_loss(
                output[..., : -self._prediction_snapshots, :],
                input[..., : -self._prediction_snapshots, :],
            )
            prediction_loss = mse_loss(
                output[..., -self._prediction_snapshots :, :],
                input[..., -self._prediction_snapshots :, :],
            )
        else:
            reconstruction_loss = mse_loss(output, input)
            prediction_loss = 0

        return (
            self._encoding_weight * decoder_loss
            + self._phase_space_weight * psp_loss
            + self._reconstruction_weight * reconstruction_loss
            + self._prediction_weight * prediction_loss
        )

    def _train_step(self, loader):
        self.train()
        loss_sum = 0.0
        for i, minibatch in enumerate(loader):
            self._optimizer.zero_grad()
            output = self(minibatch)
            loss = self._compute_loss(output, minibatch)
            loss.backward()
            self._optimizer.step()

            loss_sum += loss.item()
        return loss_sum / (i + 1)

    def _eval_step(self, loader):
        self.eval()
        return np.mean(
            [
                self._compute_loss(self(minibatch), minibatch).item()
                for minibatch in loader
            ]
        )

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
            start_training_time = time.time_ns()
            train_loss = self._train_step(train_dataloader)
            training_time = (time.time_ns() - start_training_time) / 1_000_000
            train_loss_arr.append(train_loss)

            start_eval_time = time.time_ns()
            eval_loss = self._eval_step(test_dataloader)
            eval_time = (time.time_ns() - start_eval_time) / 1_000_000
            eval_loss_arr.append(eval_loss)

            if epoch % self._print_every == 0:
                logging.info(
                    f"[{epoch}] loss: {eval_loss:.4f}, train_time: {training_time:.2f} ms, eval_time: {eval_time:.2f} ms"
                )

        np.save(f"{self._label}_train_loss.npy", train_loss_arr)
        np.save(f"{self._label}_eval_loss.npy", eval_loss_arr)

        return self
