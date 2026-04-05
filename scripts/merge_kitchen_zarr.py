#!/usr/bin/env python3
"""
Gabungkan beberapa zarr Franka Kitchen (format gen_demonstration_franka_kitchen.py)
menjadi satu replay buffer untuk training.

Contoh:
  python scripts/merge_kitchen_zarr.py \\
    --inputs data/kitchen_complete.zarr data/kitchen_partial.zarr \\
    --output_zarr data/kitchen_merged.zarr
"""

import argparse
import os
import numpy as np
import zarr


def merge_zarrs(input_paths, output_zarr, overwrite=False):
    output_zarr = os.path.abspath(os.path.expanduser(output_zarr))
    if os.path.exists(output_zarr) and not overwrite:
        raise FileExistsError(f"{output_zarr} exists; pass --overwrite")

    states, actions, pcs = [], [], []
    merged_ends = []
    offset = 0

    for p in input_paths:
        p = os.path.abspath(os.path.expanduser(p))
        if not os.path.isdir(p):
            raise FileNotFoundError(p)
        root = zarr.open_group(p, mode='r')
        s = root['data']['state'][:]
        a = root['data']['action'][:]
        pc = root['data']['point_cloud'][:]
        ee = root['meta']['episode_ends'][:]

        if s.shape[0] != a.shape[0] or s.shape[0] != pc.shape[0]:
            raise ValueError(f"Length mismatch in {p}")
        if ee.size == 0 or ee[-1] != len(s):
            raise ValueError(
                f"episode_ends inconsistent with data length in {p}: "
                f"ends[-1]={ee[-1]} len={len(s)}")

        states.append(s)
        actions.append(a)
        pcs.append(pc)
        merged_ends.extend((ee + offset).tolist())
        offset += len(s)

    all_states = np.concatenate(states, axis=0)
    all_actions = np.concatenate(actions, axis=0)
    all_pcs = np.concatenate(pcs, axis=0)
    merged_ends = np.array(merged_ends, dtype=np.int64)

    if merged_ends[-1] != len(all_states):
        raise RuntimeError("Internal merge error: episode_ends vs total steps")

    parent = os.path.dirname(output_zarr)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.exists(output_zarr) and overwrite:
        import shutil
        shutil.rmtree(output_zarr)

    store = zarr.DirectoryStore(output_zarr)
    root = zarr.group(store, overwrite=True)
    dg = root.create_group('data')
    mg = root.create_group('meta')
    n = len(all_states)
    dg.create_dataset('state', data=all_states,
                      chunks=(min(1000, n),) + all_states.shape[1:],
                      dtype=np.float32)
    dg.create_dataset('action', data=all_actions,
                      chunks=(min(1000, n),) + all_actions.shape[1:],
                      dtype=np.float32)
    dg.create_dataset('point_cloud', data=all_pcs,
                      chunks=(min(100, n),) + all_pcs.shape[1:],
                      dtype=np.float32)
    mg.create_dataset('episode_ends', data=merged_ends, dtype=np.int64)
    print(f"Wrote {output_zarr}: T={n}, episodes={len(merged_ends)}, "
          f"state={all_states.shape}, pc={all_pcs.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help='Path zarr sumber')
    ap.add_argument('--output_zarr', required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    merge_zarrs(args.inputs, args.output_zarr, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
