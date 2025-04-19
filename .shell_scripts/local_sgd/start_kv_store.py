# start_kv_store.py
import time
import os
import sys
import logging
import argparse
from datetime import timedelta
from torch.distributed import TCPStore  # Import directly

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('SimpleKVStoreDaemon')

def main():
    parser = argparse.ArgumentParser(description="Starts a simple TCPStore KV daemon.")
    parser.add_argument("--host", type=str, required=True, help="Host/IP to bind.")
    parser.add_argument("--port", type=int, required=True, help="Port to bind.")
    parser.add_argument("--timeout", type=int, default=3600, help="Store operation timeout (seconds).")
    parser.add_argument("--pid-file", type=str, default=None, help="File to write PID.")

    args = parser.parse_args()

    # Write PID file if requested
    if args.pid_file:
        try:
            pid = os.getpid()
            pid_dir = os.path.dirname(args.pid_file)
            if pid_dir and not os.path.exists(pid_dir): os.makedirs(pid_dir, exist_ok=True)
            with open(args.pid_file, 'w') as f: f.write(str(pid))
            logger.info(f"Process ID {pid} written to {args.pid_file}")
        except IOError as e:
            logger.error(f"Failed to write PID file {args.pid_file}: {e}")

    store_endpoint = f"{args.host}:{args.port}"
    logger.info(f"Initializing direct TCPStore at {store_endpoint}")
    store = None # Define before try block

    try:
        # The store needs a world_size. For a standalone daemon acting as the
        # master coordinator for get/set, world_size=1 (itself) is correct.
        # is_master=True makes it listen for incoming connections.
        store = TCPStore(
            args.host,
            args.port,
            world_size=1,
            is_master=True, # Crucial: Makes this process the listening server
            timeout=timedelta(seconds=args.timeout)
        )
        logger.info(f'Direct TCPStore started successfully at {store_endpoint}.')
        logger.info('Daemon running. Waiting for connections... (Press Ctrl+C to stop)')

        # Keep the process alive indefinitely
        while True:
            # You could potentially add checks here (e.g., check store health if API allows)
            time.sleep(300) # Sleep for 5 minutes

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt. Shutting down.")
    except Exception as e:
        # Catch potential errors during TCPStore init (e.g., port already in use)
        logger.exception(f'FATAL ERROR starting or running the store: {e}')
        sys.exit(1)
    finally:
        # TCPStore uses a __del__ method for cleanup, but explicit closing is good practice if available.
        # However, there isn't a public close() method. Relies on process termination.
        logger.info("Cleaning up PID file...")
        # Cleanup PID file
        if args.pid_file and os.path.exists(args.pid_file):
             try:
                 with open(args.pid_file, 'r') as f: stored_pid = f.read().strip()
                 if stored_pid == str(os.getpid()):
                     os.remove(args.pid_file); logger.info("PID file removed.")
                 else: logger.warning("PID mismatch, not removing PID file.")
             except Exception as e: logger.error(f"Error removing PID file: {e}")

    logger.info('Daemon process finished.')
    sys.exit(0)

if __name__ == "__main__":
    main()