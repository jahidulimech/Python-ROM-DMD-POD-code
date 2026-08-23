"""OpenFOAM ASCII I/O utilities for the ROM/DMD workflow."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable
import numpy as np

TIME_RE = re.compile(r"^[0-9.+-]+(?:[eE][+-]?[0-9]+)?$")


def numeric_time_dirs(root: Path) -> dict[float, str]:
    """Return numeric time directory names keyed by physical time."""
    out: dict[float, str] = {}
    for child in root.iterdir():
        if child.is_dir() and TIME_RE.match(child.name):
            value = float(child.name)
            if value in out:
                raise ValueError(f"Duplicate numeric time {value}: {out[value]}, {child.name}")
            out[value] = child.name
    return dict(sorted(out.items()))


def _list_payload(text: str, object_name: str) -> tuple[int, str]:
    """Extract a top-level OpenFOAM list payload by balanced parentheses."""
    marker = re.search(r"\n\s*(\d+)\s*\n\s*\(", text)
    if not marker:
        raise ValueError(f"Could not find count/list payload in {object_name}")
    count = int(marker.group(1))
    start = marker.end()
    level, end = 1, start
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            level += 1
        elif char == ")":
            level -= 1
            if level == 0:
                end = index
                break
    if level:
        raise ValueError(f"Unclosed list payload in {object_name}")
    return count, text[start:end]


def read_label_list(path: Path) -> np.ndarray:
    count, payload = _list_payload(path.read_text(encoding="utf-8", errors="replace"), str(path))
    values = np.fromstring(payload, sep=" ", dtype=int)
    if values.size != count:
        raise ValueError(f"{path}: expected {count} labels; got {values.size}")
    return values


def read_points(path: Path) -> np.ndarray:
    count, payload = _list_payload(path.read_text(encoding="utf-8", errors="replace"), str(path))
    groups = re.findall(r"\(([^()]+)\)", payload)
    values = np.array([np.fromstring(group, sep=" ", dtype=float) for group in groups], dtype=float)
    if values.shape != (count, 3):
        raise ValueError(f"{path}: expected ({count}, 3) point array; got {values.shape}")
    return values


def read_faces(path: Path) -> list[np.ndarray]:
    count, payload = _list_payload(path.read_text(encoding="utf-8", errors="replace"), str(path))
    faces = [np.fromstring(group, sep=" ", dtype=int)
             for group in re.findall(r"\d+\s*\(([^()]+)\)", payload)]
    if len(faces) != count:
        raise ValueError(f"{path}: expected {count} faces; got {len(faces)}")
    return faces


def local_cell_centres(poly_mesh: Path, n_cells: int) -> np.ndarray:
    """Compute local hexahedral cell centres from actual mesh topology."""
    points = read_points(poly_mesh / "points")
    faces = read_faces(poly_mesh / "faces")
    owner = read_label_list(poly_mesh / "owner")
    neighbour = read_label_list(poly_mesh / "neighbour")
    if owner.size != len(faces) or neighbour.size > owner.size:
        raise ValueError(f"Invalid owner/neighbour topology in {poly_mesh}")
    cell_faces: list[list[int]] = [[] for _ in range(n_cells)]
    for face_id, cell in enumerate(owner):
        cell_faces[int(cell)].append(face_id)
    for face_id, cell in enumerate(neighbour):
        cell_faces[int(cell)].append(face_id)
    centres = np.empty((n_cells, 3), dtype=float)
    for cell_id, face_ids in enumerate(cell_faces):
        if not face_ids:
            raise ValueError(f"Cell {cell_id} in {poly_mesh} has no faces")
        point_ids = np.unique(np.concatenate([faces[face_id] for face_id in face_ids]))
        centres[cell_id] = points[point_ids].mean(axis=0)
    return centres


def read_internal_field(path: Path) -> np.ndarray:
    """Read uniform or nonuniform scalar/vector internalField from ASCII OpenFOAM."""
    text = path.read_text(encoding="utf-8", errors="replace")
    field_match = re.search(r"internalField\s+(.*?)(?:;)", text, flags=re.S)
    if not field_match:
        raise ValueError(f"{path}: internalField is absent")
    declaration = field_match.group(1).strip()
    if declaration.startswith("uniform"):
        token = declaration[len("uniform"):].strip()
        vector = re.full((1, 3), np.fromstring(token.strip("()"), sep=" ", dtype=float))
        return vector if vector.shape[1] == 3 else np.array([float(token)])
    list_match = re.search(
        r"nonuniform\s+List<(?P<kind>scalar|vector)>\s*(?P<count>\d+)\s*\(\s*(?P<body>.*?)\s*\)\s*$",
        declaration, flags=re.S)
    if not list_match:
        raise ValueError(f"{path}: unsupported internalField declaration")
    count = int(list_match.group("count"))
    kind, body = list_match.group("kind"), list_match.group("body")
    if kind == "scalar":
        values = np.fromstring(body, sep=" ", dtype=float)
        if values.size != count:
            raise ValueError(f"{path}: expected {count} scalars; got {values.size}")
        return values
    groups = re.findall(r"\(([^()]+)\)", body)
    values = np.array([np.fromstring(group, sep=" ", dtype=float) for group in groups], dtype=float)
    if values.shape != (count, 3):
        raise ValueError(f"{path}: expected {count} vectors; got {values.shape}")
    return values


def processor_directories(case_dir: Path) -> list[Path]:
    procs = sorted(
        (path for path in case_dir.glob("processor*") if path.is_dir()),
        key=lambda path: int(path.name.replace("processor", "")))
    if not procs:
        raise FileNotFoundError(f"No processor directories under {case_dir}")
    expected = list(range(len(procs)))
    found = [int(path.name.replace("processor", "")) for path in procs]
    if found != expected:
        raise ValueError(f"Processor numbering is not contiguous: {found}")
    return procs


def discover_parallel_data(case_dir: Path, required_fields: Iterable[str]) -> dict:
    """Validate shared time set, cell mappings and required parallel fields."""
    procs = processor_directories(case_dir)
    per_proc_times = [numeric_time_dirs(proc) for proc in procs]
    shared_times = set(per_proc_times[0])
    for mapping in per_proc_times[1:]:
        shared_times &= set(mapping)
    if not shared_times:
        raise ValueError("No common numeric time directories across processors")
    addresses, centres = [], []
    for proc in procs:
        poly = proc / "constant" / "polyMesh"
        address = read_label_list(poly / "cellProcAddressing")
        addresses.append(address)
        centres.append(local_cell_centres(poly, address.size))
    all_address = np.concatenate(addresses)
    n_global = int(all_address.max()) + 1
    if np.any(all_address < 0) or all_address.size != n_global or set(all_address.tolist()) != set(range(n_global)):
        raise ValueError("cellProcAddressing is not a one-to-one complete global cell map")
    x = np.empty(n_global)
    for address, centre in zip(addresses, centres):
        x[address] = centre[:, 0]
    if not np.all(np.diff(x) > 0):
        order = np.argsort(x)
        if not np.array_equal(order, np.arange(n_global)):
            raise ValueError("Global cell labels are not in increasing physical-x order")
    valid_times, missing = [], {}
    for time in sorted(shared_times):
        names = [per_proc_times[i][time] for i in range(len(procs))]
        absent = []
        for field in required_fields:
            for proc, name in zip(procs, names):
                if not (proc / name / field).is_file():
                    absent.append(f"{proc.name}/{name}/{field}")
        if absent:
            missing[time] = absent
        else:
            valid_times.append(time)
    return {
        "processors": procs, "time_names": per_proc_times, "times": np.array(valid_times, dtype=float),
        "missing": missing, "addresses": addresses, "x": x, "n_cells": n_global,
    }


def reconstruct_field(info: dict, time: float, field: str) -> np.ndarray:
    """Map per-processor field arrays to global cells using cellProcAddressing."""
    n_cells = info["n_cells"]
    out: np.ndarray | None = None
    for proc, mapping, address in zip(info["processors"], info["time_names"], info["addresses"]):
        path = proc / mapping[float(time)] / field
        values = read_internal_field(path)
        if values.shape[0] == 1:  # uniform field: expand to local mesh extent
            values = np.repeat(values, address.size, axis=0)
        if values.shape[0] != address.size:
            raise ValueError(f"{path}: {values.shape[0]} values; mapping has {address.size} cells")
        if out is None:
            out = np.empty((n_cells,) + values.shape[1:], dtype=float)
        out[address] = values
    if out is None or not np.isfinite(out).all():
        raise ValueError(f"Failed reconstruction or nonfinite values: field={field}, time={time}")
    return out
