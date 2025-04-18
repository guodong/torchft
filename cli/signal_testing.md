# Testing Signal Handling with TorchFT CLI

This document explains how to use the enhanced TorchFT CLI to test signal handling capabilities.

## Overview

The TorchFT CLI has been enhanced to support testing signal handling functionality, similar to how the Manager handles errors in distributed training. When enabled, error messages received through the ErrorBus will trigger a SIGUSR1 signal, which is then handled by a registered signal handler.

## Usage

### Starting the Error Bus Server

First, start the error bus server in a separate terminal:

```bash
python -m torchft.distributed.messaging.error_bus master
```

### Running the TorchFT CLI with Signal Testing

To run the TorchFT CLI with signal testing enabled, use the `-s` or `--test-signals` flag:

```bash
python -m torchft.cli.torchft_shell -H 127.0.0.1 -p 22223 -s
```

This will start the CLI with signal handling enabled. When an error message is received, it will:
1. Store the error message
2. Send a SIGUSR1 signal to the current process
3. The signal handler will process the message and simulate aborting the step

### Available Commands

The following commands are available in the CLI:

#### Basic Commands

- `help` - Show available commands
- `quit` (or `exit`, `q`) - Exit the CLI

#### Error Broadcasting Commands

- `broadcast <gpu_index> [reason]` - Send a GPU error message
  - Example: `broadcast 0 Out of memory`

- `broadcast_node <hostname> [reason]` - Send a node error message
  - Example: `broadcast_node localhost Node crashed`

- `broadcast_general [reason]` - Send a general error message
  - Example: `broadcast_general Training process failed`

#### Listening Command

- `listen` - Start listening for error messages
  - In signal testing mode, this will set up a callback that triggers the signal handler

## Testing Workflow

To test signal handling:

1. Start the error bus server (in terminal 1):
   ```bash
   python -m torchft.distributed.messaging.error_bus master
   ```

2. Start the CLI with signal testing enabled (in terminal 2):
   ```bash
   python -m torchft.cli.torchft_shell -H 127.0.0.1 -p 22223 -s
   ```

3. Start listening for messages:
   ```
   torchft> listen
   ```

4. Start another CLI instance without signal testing (in terminal 3):
   ```bash
   python -m torchft.cli.torchft_shell -H 127.0.0.1 -p 22223
   ```

5. Send an error message from the second CLI:
   ```
   torchft> broadcast 0 GPU out of memory
   ```

6. Observe in the first CLI that:
   - The error message is received
   - A SIGUSR1 signal is triggered
   - The signal handler processes the message and simulates aborting the step

## Relationship to Manager Implementation

This test functionality mirrors how the TorchFT Manager handles errors in distributed training:

1. The `_error_bus_callback` method receives an error message
2. It stores the message and sends a SIGUSR1 signal to itself
3. The signal handler calls `_abort_step()` to handle the error
4. `_abort_step()` examines the error and takes appropriate action

Testing with this enhanced CLI helps verify that the signal handling mechanism works correctly, which is essential for reliable error handling in distributed training. 