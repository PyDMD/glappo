import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from dldmd import DLDMD

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train_dldmd(
    dldmd,
    training_data,
    test_data,
    batch_size=256,
    epochs=1000,
    max_loss=0,
    print_prediction_loss=True,
    print_every=True,
    label="dldmd",
    eval_on_cpu=True,
):

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train_loss_arr = []
    eval_loss_arr = []

    if torch.cuda.is_available():
        dldmd.to("cuda", dtype=training_data.dtype)
    else:
        dldmd.to(dtype=training_data.dtype)

    for epoch in range(1, epochs + 1):
        training_time, train_loss = dldmd.train_step(train_dataloader)
        train_loss_arr.append(train_loss)

        model_label = f"{label}_e{epoch}"
        dldmd.save_model(model_label)
        eval_model = DLDMD.load_model_for_eval(model_label, eval_on_cpu)

        eval = eval_model.eval_step(test_dataloader)
        eval_time, (eval_loss, prediction_loss) = eval

        if not eval_loss_arr or min(eval_loss_arr) > eval_loss:
            dldmd.save_model(f"{label}_best")
        eval_loss_arr.append(eval_loss)

        if epoch % print_every == 0:
            logger.info(
                f"[{epoch}] loss: {eval_loss:.7f}, train_time: {training_time:.2f} ms, eval_time: {eval_time:.2f} ms"
            )
            if print_prediction_loss:
                logger.info(f"[{epoch}] prediction loss: {prediction_loss:.7f}")
        else:
            os.remove(model_label + ".pl")

        if eval_loss < max_loss:
            break

    np.save(f"{label}_train_loss.npy", train_loss_arr)
    np.save(f"{label}_eval_loss.npy", eval_loss_arr)

    return dldmd
