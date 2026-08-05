"""FedOps federated-client application entrypoint.

The filename is part of the Silo Baseline contract.  Runtime integration is
provided by FedOps instead of exposing a Flower ``ClientApp``.
"""

from .client_main import main


__all__ = ["main"]


if __name__ == "__main__":
    main()
