"""
utils.py
========
Shared helpers for logging, timing, JSON I/O and safe array normalization.
"""
from __future__ import annotations
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator
import numpy as np
from config import LOG_FORMAT, LOG_LEVEL

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
    return logger

@contextmanager
def timer(logger: logging.Logger, label: str) -> Iterator[Dict[str, float]]:
    result: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["seconds"] = elapsed
        logger.info("%s took %.3fs", label, elapsed)

def min_max_scale(values: np.ndarray, ref_min: float | None = None, ref_max: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = values.min() if ref_min is None else ref_min
    hi = values.max() if ref_max is None else ref_max
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
