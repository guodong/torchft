# FFT Design Document

## High Level Overview
FFT is the fault tolerance layer on top of large scale machine learning workloads. It implements a common protocol for different monitoring systems and processing elements to communicate with each other for fast fault tolerance. It aims for the lowest performance envelope and recovery latency. 

### Guarantees
FFT uses global step as a unit of fault tolerance, and ensures **safety** by 

a) Guaranteeing that all parameters are updated once and only once per local step.
b) It constructs a data flow graph, and guarantees that the step count of parameters later on in the data flow graph is less than or equal to parameters earlier on in the data flow graph.
c) Further, the user is able to specify the number of steps an earlier parameter group can go beyond a later group.

Other guarantees specific to ML workloads (using the language of torch.distributed),  specifically to Data Parallelism and Sharding (note that other parallelisms can be expressed in terms of data prallelism, sharding, and the data flow graphs):

d) It defines replicate groups. The user can define the sync policy for each replicate group. This could be "Per Step", "Per 3 Steps", "Asynchronous". Per Step would mean that after every local step, members of the replicate group communicate, and guarantee that all members will end up at the same state in the next commit. (Note this doesn't mean that they have to synchronize at the beginning, only at the end.)
e) It defines shard groups. A shard group is thought of one unit of fault tolerance. At the beginning of each step's computation, they have to have the same step.

(TODO: Am I missing anything? Can my language be better?)

FFT attempts to ensure **Liveness** (though makes no guarantees (?)) by retry upon fault, and when continuously faulty, start an (TODO) active failure identification procedure (similar to dlRover).

### FT Procedure

FFT splits fault tolerance into two parts:

FT1: Fault Detection
FT2: Fault Recovery

### FT1
For FT1, FFT integrates the existing distributed training runtimes (train.py processes) with an error_bus that provides the runtimes the ability to broadcast local errors to each other, and receive error messages from its peers as well as a centralized fault monitoring system.

Then, these error messages are delivered to its peers in two ways.

1. Python signalling
2. pytorch.futures.set_exception()

We implement the second way as python signaling is unable to interrupt blocking calls in C, such as pytorch.futures.wait(). However, a benefit of using python signaling is that it is generally very safe. We ensure that python signal handler coorporates with the pytorch.futures.set_exception signal handling methods through a conditional variable, so the program rolls back successfully even when both methods are implemented.

Note: The fault monitoring system can be currently simulated by a shell CLI that can send error messages on the error bus.

### FT2

For Fault Recovery, FFT defines a semantics for roll-back based on transaction semantics. Currently, we plan to first implement this in pytorch.distributed, in an Hybrid Sharded Data Parallel setting before generalizing to general transaction semantics.

However, as a general overview, we roll back in the following way:

1. Run-time register the latest available valid state.
2. After changing its state, the run-time updates its state.
3. Whenever before commiting a change to its state, the run-time checks that the change is valid.
4. Upon failure, the run-time rolls back to the previous valid state. Note that this state may be distributes across several machines, so that the different processes in the run-time needs to communicate with each other to ensure that all has rolled back collectively to a globally valid state before continuing. Note that states may change after roll back (e.g. the total number of processes that are alive).

In the context of Hybrid Sharded Data Parallel:

A locally valid state is a valid set of model weights, optimizer states, and statefuldataloader states. Here, valid means that this is right after a successful gradient update. This state is indexed by a "step" variable. This can be captured by pytorch's load_state_dict and save_state_dict. This ensures the validity of the FSDP part of the run-time, since to have a valid set of model weights, optimizer states, and dataloader states require

A globally valid state requires that all the processes participating in the the data-parallel communication groups in HSDP have the same step (assuming that each process has a valid state, if they have the same step they have the same state).

A change in state is an optimizer.step(). The run-time checks that all participating processes are in the correct state (have finished forward and backward pass successfully) before doing optimizer.step().

We roll back by restarting the forward and backward pass.

Observe that here the validity of the state comes from the success of all-reduce of the gradients. This is because the gradient is the only distributed state in HSDP. After all-reduce, the global gradient is replicated across each runtime and is not distributed. Therefore, one can commit (optimizer.step()) without communicating to other processes.

### Finite State machine

The FFT programming model is based on a finite state machine.

For each programming paradigm, we need to define the transition table and the exit guard. 

```
                +-------------+
                |  INIT       |
                +-------------+
                        |
                        | quorum ready
                        v
      +-----------+   compute grads   +-------------+
      |  NORMAL   | ───────────────► |  PREPARED   |
      +-----------+                  +-------------+
          ^  |                           |    |
          |  | (local/remote error)      |    | all peers vote yes
          |  |                           |    v
          |  |                           | +-----------+
          |  |                           | |  COMMIT   |
          |  |                           | +-----------+
          |  |                           |     |
          |  | (vote no/err)             |     | step++
          |  |                           |     v
          |  |                           | +-----------+
          |  |                           | | DRAIN     | (optional)
          |  |                           | +-----------+
          |  |                           |     |
          |  |                           |     v
          |  +---------------------------------+
          |                                    |
          |  rollback_req from any peer        |
          v                                    |
 +---------------+                             |
 | ROLLBACK_REQ  |<----------------------------+
 +---------------+
          |
          | recv ROLLBACK_REQ from quorum
          v
 +---------------+
 | ROLLBACK_WAIT |
 +---------------+
          |
          | all ACKs exchanged,
          | PG abort + futures cancel
          v
 +---------------+
 |   HEALING     |
 +---------------+
          | ckpt applied + quorum OK
          v
      +-----------+
      |  NORMAL   |
      +-----------+
```

## Current Implementation

#### `error_bus`

Error bus is the transport layer for fault tolerance related messaging. It implements an interface for a FIFO Message_Queue with global puts and local gets.

Whenever the error_bus is initialized locally, a `listening_thread` is initialized that listens to incoming messages at `{host_name}:{port}`.

The `listening_thread` constantly monitors the message_queue, pops messages from it (`recv`), and executes a custom defined callback.

The `broadcast` puts a message onto the message queue for others to consume. 

#### Note on current error_bus implementation:

The current error_bus backend uses a torch.distributed.TCPStore. This is a distributed KV store. We implement a custom compare-set based algorithm with two TCPStore connections to implement a FIFO message queue system where every host puts messages onto the queue for everyone else to see, but pops messages from the queue only for itself. This part of the design is subject to change.

The message queue can be enhanced. One idea is to employ Kafka-like producer-consumer semantics with topics. This will enable much more generality to our system. One potentially useful backend that we could use is NATS: https://docs.nats.io/.

#### Error Bus Workflow
error_bus (broadcast(messages)) 
-> (message = self.recv() ) train.error_bus.listening_thread 
-> (message response) train.error_bus.callback(message)

#### FFT Module Integration

The callback of the error_bus is defined by the Fast Fault Tolerance protocol. The FFT protocol does a few things:

1. it checks whether the error message is in the FFT required formats. If it is not, then it alerts the user through a log message, but does not abort the runtime.

### `ManagerFFT`

ManagerFFT:

ManagerFFT enhances `torchft.Manager` by integrating it with the `error_bus`. It does so through the `FFT` module.

ManagerFFT implements the following additional methods:

```python
class ManagerFFT(torchft.manager):
    super.__init__(...)

    def run_error_bus():

    def report_error(self, e: Exception) -> None:
        """
        Report an error to the manager.

        At the same time, format the error into a FFT.GPUErrorMessage format and broadcast it.

        This will cause the manager to skip the current step and will be
        reconfigured on the next step.

        This should be called when an error occurs that leads to a corrupted
        gradient that needs to be discarded.
        """
        super.report_error(e)
        message = self.FFT.format_GPU_message(e, self._rank) # Could be GPU index or something else
        self.eb.broadcast(message)

class FFT:
    """
    stateless class. Consists of pure functions. Defines a namespace of functions.
    """
    def format_message(e, rank):

        
```

### `FFT`

The FFT module defines the FFT protocol. The protocol implementation may be specific to the language model training library. The FFT protocol is currently integrated into pytorch.distributed ecosystem. Thus, implicitly, the FFT module called here is FFT.pytorchn subset of the general FFT.

FFT protocol relies on the definition of transaction semantics. Whenever there is a failure, it rolls back to the last valid state. This may require alerting other training processes in the distributed training system.

It employs the following abstractions:

`FFT.`

It encompasses the following scope:

`FFT.register_signal_handler` defines how the signal handling is responded. This logic is specific to python and uses the python signal handling.

`FFT.on_message_callback(message: FFTErrorMessage, config: FFTConfig)` defines how 

The `FFT` module does the following:

The `FFT` module registers the signal handlers to be used in `register_signal_handler`

At the same time. 

```python
import signal

class FFT:
    def __init__(
        self,

    )

    def register_signal_handlers(
        self
    )

```

## TorchFT

Description of TorchFT can be found on the github.

## TODOs

- Defining transaction semantics with FFT
- Build out an explicit data flow graph with FFT
- Active failure injection procedure
- TCPStore key space wrap around (need to increment prefix?)
- TCPStore compaction
- Drain (gradeful exit of any one rank that prevenst others from receiving a fault)


- TODO: FSM

                         ┌──────────────┐
                         │     INIT     │
                         └──────────────┘
                                |
                                |  quorum ready
                                v
                  (step = s) ┌──────────────┐
           ┌─────────────────│   NORMAL     │◄──────────────────┐
           │                 └──────────────┘                   │
           │                      |  backward finished          │
           │                      v                             │
           │            ┌─────────────────────┐                 │
           │            │      PREPARED       │                 │
           │            └─────────────────────┘                 │
           │           /|  vote=yes + !want_drain   \           │
           │          / |                            \          │
           │ vote=no /  | vote=yes + want_drain       \         │
           │  or    /   |                              \        │
   Fault1/2│ error /    v                               v       │
           ▼       ┌────────────┐              ┌──────────────────┐
   ┌───────────────│ COMMIT(s) │──────────────│  DRAIN_COMMIT(s) │
   │               └────────────┘              └──────────────────┘
   │                    | step++                               |
   │                    |                                      |  broadcast DRAIN_NOTICE
   │                    v                                      v
   │               ┌────────────┐                       ┌─────────────┐
   │               │ NORMAL(s+1)│                       │ DRAIN_WAIT  │
   │               └────────────┘                       └─────────────┘
   │                                                         |
   │                           all DRAIN_ACK                 |
   │                                                         v
   │                                                   ┌───────────┐
   │                                                   │  EXIT     │
   │                                                   └───────────┘
   │
   │ rollback_req broadcast
   ▼
┌────────────────┐
│ ROLLBACK_REQ   │  (collect rollback requests from all participants)
└────────────────┘
         |
         | quorum_seen_all(req)   ── broadcast ROLLBACK_ACK
         v
┌────────────────┐
│ ROLLBACK_SYNC  │  (barrier: pg.abort, cancel futures, zero grads)
└────────────────┘
         |
         | if Fault1  → fetch & apply ckpt
         | if Fault2  → skip fetch (local state OK)
         v
┌────────────────┐
│    HEALING     │  (pg.configure(), wait for quorum OK)
└────────────────┘
         |
         v
     NORMAL
