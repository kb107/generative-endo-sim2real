"""DataLoader worker lifecycle helpers."""

import multiprocessing.resource_sharer as resource_sharer
import multiprocessing.util as multiprocessing_util


def configure_worker_cleanup(_: int) -> None:
    """Close resource-sharing sockets before removing NFS temporary files."""
    multiprocessing_util.Finalize(
        None,
        resource_sharer.stop,
        exitpriority=10,
    )
