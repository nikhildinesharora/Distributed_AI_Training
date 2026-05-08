from __future__ import annotations

import ipaddress
import os
import platform
import queue
import re
import shutil
import subprocess
import threading
import time
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import CifarCnn

DEFAULT_DIST_TIMEOUT_SECONDS = 60.0
TAILSCALE_IPV4_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def _run_text(command: list[str], timeout: float = 3.0) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _is_tailscale_ipv4(address: str) -> bool:
    try:
        return ipaddress.ip_address(address) in TAILSCALE_IPV4_NETWORK
    except ValueError:
        return False


def _interface_for_darwin_route(address: str) -> str:
    output = _run_text(["route", "-n", "get", address])
    match = re.search(r"^\s*interface:\s*(\S+)", output, flags=re.MULTILINE)
    return match.group(1) if match else ""


def _tailscale_interface_from_ifconfig() -> str:
    output = _run_text(["ifconfig"])
    current = ""
    for line in output.splitlines():
        header = re.match(r"^([A-Za-z0-9_.-]+):\s", line)
        if header:
            current = header.group(1)
            continue
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "inet" and _is_tailscale_ipv4(parts[1]):
            return current
    return ""


def _windows_tailscale_interface() -> str:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return ""
    command = (
        "$ip = Get-NetIPAddress -AddressFamily IPv4 "
        "| Where-Object { $_.IPAddress -like '100.*' } "
        "| Select-Object -First 1; "
        "if ($ip) { $ip.InterfaceAlias }"
    )
    return _run_text([powershell, "-NoProfile", "-Command", command])


def _linux_route_interface(address: str) -> str:
    output = _run_text(["ip", "route", "get", address])
    parts = output.split()
    if "dev" in parts:
        index = parts.index("dev") + 1
        if index < len(parts):
            return parts[index]
    return ""


def _detect_gloo_socket_ifname(master_addr: str) -> str:
    system = platform.system()
    if system == "Darwin":
        return _interface_for_darwin_route(master_addr) or _tailscale_interface_from_ifconfig()
    if system == "Windows":
        return _windows_tailscale_interface()
    return _linux_route_interface(master_addr)


def _configure_gloo_network(master_addr: str, rank: int) -> None:
    if os.environ.get("GLOO_SOCKET_IFNAME"):
        return
    if not _is_tailscale_ipv4(master_addr):
        return
    interface = _detect_gloo_socket_ifname(master_addr)
    if interface:
        os.environ["GLOO_SOCKET_IFNAME"] = interface
        print(
            f"[distributed] rank {rank}: using GLOO_SOCKET_IFNAME={interface}",
            flush=True,
        )


def choose_device(accelerator: str) -> torch.device:
    requested = accelerator.lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if requested == "mps" and mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _allreduce_grads(model: CifarCnn, world_size: int, device: torch.device) -> None:
    params = [param for param in model.parameters() if param.grad is not None]
    if not params:
        return

    flat = torch.cat([param.grad.detach().cpu().view(-1) for param in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size

    offset = 0
    for param in params:
        size = param.grad.numel()
        param.grad.copy_(flat[offset : offset + size].view_as(param.grad).to(device))
        offset += size


def _build_loader(
    shard: tuple[Any, Any],
    batch_size: int,
    batches_per_epoch: int,
) -> DataLoader:
    images_np, labels_np = shard
    images = torch.from_numpy(images_np.copy()).float().permute(0, 3, 1, 2) / 255.0
    labels = torch.from_numpy(labels_np.copy()).long()
    usable = min(len(images), max(1, batches_per_epoch) * max(1, batch_size))
    return DataLoader(
        TensorDataset(images[:usable], labels[:usable]),
        batch_size=max(1, batch_size),
        shuffle=True,
        drop_last=True,
    )


def _broadcast_model(model: CifarCnn, device: torch.device) -> None:
    for parameter in model.parameters():
        buffer = parameter.data.detach().cpu().clone().contiguous()
        dist.broadcast(buffer, src=0)
        parameter.data.copy_(buffer.to(device))


def _run_epoch(
    model: CifarCnn,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    world_size: int,
    device: torch.device,
    batches_per_epoch: int,
    stop_event: threading.Event,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    batch_count = 0

    for images_batch, labels_batch in loader:
        if batch_count >= batches_per_epoch or stop_event.is_set():
            break
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images_batch), labels_batch)
        loss.backward()
        _allreduce_grads(model, world_size, device)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batch_count += 1

    local_loss = total_loss / max(1, batch_count)
    loss_tensor = torch.tensor([local_loss], dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return {
        "loss": float((loss_tensor / world_size).item()),
        "batches": batch_count,
        "device": str(device),
    }


def run_training(
    rank: int,
    world_size: int,
    master_addr: str,
    dist_port: int,
    shard: tuple[Any, Any],
    accelerator: str,
    batch_size: int,
    batches_per_epoch: int,
    epochs: int,
    result_q: queue.SimpleQueue[dict[str, Any]],
    stop_event: threading.Event | None = None,
    timeout_seconds: float = DEFAULT_DIST_TIMEOUT_SECONDS,
) -> None:
    stop = stop_event or threading.Event()
    device = choose_device(accelerator)
    timeout = timedelta(seconds=max(1.0, timeout_seconds))
    current_epoch = 0
    _configure_gloo_network(master_addr, rank)

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{dist_port}",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
    except Exception as exc:
        result_q.put(
            {
                "type": "distributed_error",
                "epoch": 0,
                "rank": rank,
                "error": f"dist init: {exc}",
            }
        )
        return

    try:
        torch.manual_seed(42)
        model = CifarCnn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = _build_loader(shard, batch_size, batches_per_epoch)

        _broadcast_model(model, device)
        dist.barrier()

        for epoch in range(1, epochs + 1):
            current_epoch = epoch
            if stop.is_set():
                result_q.put(
                    {
                        "type": "distributed_error",
                        "epoch": epoch,
                        "rank": rank,
                        "error": "training stopped",
                    }
                )
                return

            epoch_started = time.monotonic()
            result = _run_epoch(
                model=model,
                optimizer=optimizer,
                loader=loader,
                world_size=world_size,
                device=device,
                batches_per_epoch=batches_per_epoch,
                stop_event=stop,
            )
            dist.barrier()
            duration_seconds = time.monotonic() - epoch_started
            result_q.put(
                {
                    "type": "distributed_epoch",
                    "epoch": epoch,
                    "rank": rank,
                    "loss": float(result["loss"]),
                    "batches": int(result["batches"]),
                    "device": str(result["device"]),
                    "duration_seconds": duration_seconds,
                }
            )

        dist.barrier()
        result_q.put({"type": "distributed_complete", "rank": rank, "epochs": epochs})
    except Exception as exc:
        result_q.put(
            {
                "type": "distributed_error",
                "epoch": current_epoch,
                "rank": rank,
                "error": str(exc),
            }
        )
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
