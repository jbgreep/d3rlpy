__all__ = ["run_healthcheck"]


def run_healthcheck() -> None:
    _check_gym()
    _check_pytorch()


def _check_gym() -> None:
    import gymnasium

    if gymnasium.__version__ < "1.0.0":
        raise ValueError(
            "Gymnasium version is too outdated. "
            "Please upgrade Gymnasium to 1.0.0 or later."
        )


def _check_pytorch() -> None:
    import torch

    if torch.__version__ < "2.5.0":
        raise ValueError(
            "PyTorch version is too outdated. "
            "Please upgrade PyTorch to 2.5.0 or later."
        )
