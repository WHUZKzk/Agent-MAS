"""
BaseAgent — shared state tracking and logging for all AutoSR agents.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    name: str
    status: str          # "ok" | "failed" | "skipped"
    elapsed_s: float
    detail: Optional[str] = None


@dataclass
class AgentState:
    steps: List[StepRecord] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record(self, name: str, status: str, elapsed_s: float, detail: str = None):
        self.steps.append(StepRecord(name=name, status=status, elapsed_s=elapsed_s, detail=detail))

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_elapsed_s": round(self.elapsed, 2),
            "steps": [
                {
                    "name": s.name,
                    "status": s.status,
                    "elapsed_s": round(s.elapsed_s, 2),
                    **({"detail": s.detail} if s.detail else {}),
                }
                for s in self.steps
            ],
        }


class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.state = AgentState()
        self.logger = logging.getLogger(f"autosr.agents.{name}")

    def reset(self):
        self.state = AgentState()

    def _run_step(self, step_name: str, fn, *args, **kwargs):
        """
        Execute fn(*args, **kwargs), record timing + outcome, propagate exceptions.
        Use this wrapper for every logical step inside an agent's run() method.
        """
        self.logger.info("[%s] → %s", self.name, step_name)
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            self.state.record(step_name, "ok", elapsed)
            self.logger.info("[%s] ✓ %s (%.1fs)", self.name, step_name, elapsed)
            return result
        except Exception as exc:
            elapsed = time.time() - t0
            self.state.record(step_name, "failed", elapsed, detail=str(exc))
            self.logger.error("[%s] ✗ %s: %s", self.name, step_name, exc)
            raise
