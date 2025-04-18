from unittest import TestCase


class TestErrorBus(TestCase):
    def test_error_bus(self) -> None:
        # Test that the error bus is initialized correctly
        from torchft.distributed.messaging.error_bus import ErrorBus

        error_bus = ErrorBus()
        self.assertIsNotNone(error_bus)