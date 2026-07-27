"""Flower ClientApp placeholder.

FedOps participation is launched by ``launcher_app`` as two local processes.
The ClientApp remains present so the project is a complete Flower App bundle.
"""

from flwr.clientapp import ClientApp


app = ClientApp()
