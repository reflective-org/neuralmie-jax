r"""Convert NeuralMie Fortran-Keras-Bridge ``.txt`` weights to ``.npz``.

Derived from pnnl/NEURALMIE @ 0b5b1d8 (BSD-2-Clause,
Copyright 2024 Battelle Memorial Institute).

The FKB text format, as shipped upstream (23 lines, tab-delimited, ``%.7e``)::

    line 1      : 11    -- count of following header lines, NOT the layer count
    lines 2-12  : ``input\t4`` / ``dense\t69`` / ``swish\t0`` / ... / ``linear\t0``
    line 13     : the final Adam learning rate -- NOT a weight; skipped
    lines 14-18 : biases,  one line per Dense layer
    lines 19-23 : weights, one line per Dense layer

Each weight line is **column-major over the Keras ``(n_in, n_out)`` kernel**
(the input index varies fastest), so the kernel is rebuilt as
``vals.reshape(n_out, n_in).T``. Getting this wrong yields plausible but
incorrect outputs rather than an error.

Usage::

    python tools/convert_weights.py --upstream third_party/NEURALMIE \\
        --out src/neuralmie_jax/data [--verify]
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

import numpy as np

FORMAT_VERSION = 1

#: Expected architecture per network, as a guard against a mis-parsed header.
EXPECTED = {
    "sphere": ((4, 69, 69, 69, 69, 3), 15_045),
    "coreshell": ((7, 112, 112, 112, 112, 3), 39_203),
}


def parse_fkb(path: pathlib.Path) -> tuple[tuple[int, ...], list[str], list[np.ndarray], list[np.ndarray]]:
    """Parse one FKB ``.txt`` file.

    Args:
        path: the ``.txt`` file to read.

    Returns
    -------
        ``(layer_sizes, activations, kernels, biases)``, with each kernel in
        Keras ``(n_in, n_out)`` orientation so that ``y = x @ kernel + bias``.

    Raises
    ------
        ValueError: if the header, element counts or parameter total disagree
            with the parsed architecture.
    """
    lines = path.read_text().split("\n")
    n_header = int(lines[0].split()[0])
    spec = [line.split() for line in lines[1 : 1 + n_header]]

    layer_sizes = tuple(int(tok[1]) for tok in spec if tok[0] in ("input", "dense"))
    activations = [tok[0] for tok in spec if tok[0] not in ("input", "dense")]
    n_layers = len(layer_sizes) - 1
    if len(activations) != n_layers:
        raise ValueError(f"{path.name}: {n_layers} layers but {len(activations)} activations")

    # Skip the learning-rate scalar that follows the header block.
    body = lines[1 + n_header + 1 :]
    biases = [np.fromstring(body[i], sep="\t") for i in range(n_layers)]
    flat = [np.fromstring(body[n_layers + i], sep="\t") for i in range(n_layers)]

    kernels = []
    for i, vals in enumerate(flat):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        if vals.size != n_in * n_out:
            raise ValueError(
                f"{path.name} layer {i}: expected {n_in * n_out} weights, got {vals.size}"
            )
        if biases[i].size != n_out:
            raise ValueError(
                f"{path.name} layer {i}: expected {n_out} biases, got {biases[i].size}"
            )
        kernels.append(vals.reshape(n_out, n_in).T)

    return layer_sizes, activations, kernels, biases


def convert(name: str, src: pathlib.Path, dst: pathlib.Path) -> dict:
    """Convert one network and write its ``.npz``. Returns a summary dict."""
    layer_sizes, activations, kernels, biases = parse_fkb(src)

    want_sizes, want_params = EXPECTED[name]
    if layer_sizes != want_sizes:
        raise ValueError(f"{name}: layer sizes {layer_sizes} != expected {want_sizes}")
    n_params = sum(k.size for k in kernels) + sum(b.size for b in biases)
    if n_params != want_params:
        raise ValueError(f"{name}: {n_params} parameters != expected {want_params}")
    for arr in (*kernels, *biases):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: non-finite parameter")

    payload: dict[str, np.ndarray] = {
        "format_version": np.int32(FORMAT_VERSION),
        "layer_sizes": np.asarray(layer_sizes, np.int32),
        "activations": np.asarray(activations),
    }
    for i, (k, b) in enumerate(zip(kernels, biases, strict=True)):
        payload[f"kernel_{i}"] = k.astype(np.float32)
        payload[f"bias_{i}"] = b.astype(np.float32)

    # Uncompressed: float32 weights barely deflate, and this avoids a zlib
    # dependency. Note np.savez embeds zip mtimes, so the file is not
    # byte-reproducible -- CI must compare arrays, not hashes.
    np.savez(dst, **payload)
    return {
        "name": name,
        "layer_sizes": layer_sizes,
        "activations": activations,
        "n_params": n_params,
        "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "kernel0_sha256": hashlib.sha256(kernels[0].astype(np.float32).tobytes()).hexdigest(),
        "bytes": dst.stat().st_size,
    }


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", type=pathlib.Path, default=pathlib.Path("third_party/NEURALMIE"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("src/neuralmie_jax/data"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        info = convert(name, args.upstream / f"{name}.txt", args.out / f"{name}.npz")
        print(
            f"{info['name']:>10}: {info['layer_sizes']} "
            f"{info['activations']} {info['n_params']:,} params "
            f"-> {info['bytes'] / 1024:.0f} KB"
        )
        print(f"{'':>10}  kernel_0 sha256 {info['kernel0_sha256'][:16]}...")


if __name__ == "__main__":
    main()
