from argparse import Namespace

import click
import torch
from loguru import logger

from models.classifier import load_model


@click.command()
@click.option("--lr", default=0.01, type=float, help="learning rate")
@click.option(
    "--model",
    default="rand-mat-res-mnist",
    type=click.Choice(["rand-mat-res-mnist"]),
    help="model",
)
@click.option("--res-dim", default=64, type=int, help="model")
@click.option(
    "--device",
    default="mps",
    type=click.Choice(["cuda", "cpu", "mps"]),
    help="device to use for computations",
)
def main(**kwargs):
    args = Namespace(**kwargs)
    model = load_model(args)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    logger.debug(f"{args=}")
    logger.debug(f"{model=}")
    logger.debug(f"{optimizer=}")


if __name__ == "__main__":
    main()
