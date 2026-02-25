"""fsm-rs: Rust port of Essery's FSM1 (Factorial Snow Model)."""

from fsm_rs._fsm_rs import run_fsm1_batch, state_size
from fsm_rs.wrapper import run_fsm1

__all__ = ["run_fsm1_batch", "run_fsm1", "state_size"]
__version__ = "0.1.0"
