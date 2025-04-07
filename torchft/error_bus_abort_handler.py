from torchft.error_bus import Message as ErrorBusMessage
import logging

logger: logging.Logger = logging.getLogger(__name__)

def error_bus_abort_handler(pg, error_msg: ErrorBusMessage) -> None:
    logger.error(f"Aborting local process for message: {error_msg}")
    return