import argparse
import subprocess
import time
from typing import List, Tuple, Optional

import psutil

try:
    import torch
except Exception:
    torch = None


def _gpu_usage() -> Optional[float]:
    """Return current GPU memory utilisation as a fraction."""
    if (
        torch is None
        or not getattr(torch, "cuda", None)
        or not torch.cuda.is_available()
    ):
        return None
    try:
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return used / float(total)
    except Exception:
        return None


def _system_memory_usage() -> float:
    """Return system RAM usage as a fraction."""
    mem = psutil.virtual_memory()
    return mem.percent / 100.0


class GPUJobScheduler:
    """Simple GPU job scheduler based on memory usage."""

    def __init__(
        self,
        max_fraction: float = 0.9,
        default_job_fraction: float = 0.5,
        check_interval: float = 1.0,
    ) -> None:
        self.max_fraction = max_fraction
        self.default_job_fraction = default_job_fraction
        self.check_interval = check_interval
        self.pending: List[Tuple[List[str], float]] = []
        self.running: List[subprocess.Popen] = []

    def add_job(self, cmd: List[str], mem_fraction: Optional[float] = None) -> None:
        self.pending.append((cmd, mem_fraction or self.default_job_fraction))

    def _current_usage(self) -> float:
        gpu = _gpu_usage()
        if gpu is not None:
            return gpu
        return _system_memory_usage()

    def _cleanup_finished(self) -> None:
        self.running = [p for p in self.running if p.poll() is None]

    def run(self) -> None:
        """Run jobs while respecting memory limits."""
        while self.pending or self.running:
            self._cleanup_finished()
            usage = self._current_usage()
            available = self.max_fraction - usage

            i = 0
            while i < len(self.pending):
                cmd, need = self.pending[i]
                if need <= available:
                    self.running.append(subprocess.Popen(cmd))
                    available -= need
                    self.pending.pop(i)
                else:
                    i += 1
            time.sleep(self.check_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schedule jobs based on GPU memory utilisation"
    )
    parser.add_argument("script", help="Python script to run for each job")
    parser.add_argument(
        "--num_jobs", type=int, default=1, help="Number of jobs to schedule"
    )
    parser.add_argument(
        "--job_memory",
        type=float,
        default=0.5,
        help="Estimated memory fraction each job requires",
    )
    parser.add_argument(
        "--max_fraction",
        type=float,
        default=0.9,
        help="Maximum allowed memory utilisation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scheduler = GPUJobScheduler(
        max_fraction=args.max_fraction, default_job_fraction=args.job_memory
    )
    for _ in range(args.num_jobs):
        scheduler.add_job(["python", args.script])
    scheduler.run()


if __name__ == "__main__":
    main()
