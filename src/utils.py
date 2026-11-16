"""Small shared helpers."""
import platform
import random

import numpy as np

SEED = 42


def set_seed(seed=SEED):
    """Seed python, numpy and (if installed) torch (PLAN §6)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def hardware():
    """CPU model, cores, RAM and OS of this machine (PLAN §6, for timing tables)."""
    cpu = platform.processor() or "unknown"
    ram_gb = float("nan")
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                ram_gb = round(int(line.split()[1]) / 1024 ** 2, 1)
                break
    except OSError:
        pass
    import os
    return {"cpu": cpu, "cores": os.cpu_count(), "ram_gb": ram_gb,
            "os": platform.platform(), "python": platform.python_version()}
