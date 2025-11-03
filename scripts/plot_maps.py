#!/usr/bin/env python3
"""
Map plotting utility
- Scans the `map/` directory for subfolders containing `node.csv` and `edge.csv`.
- Reads nodes (expects columns: id,x,y or similar) and edges (expects from,to) and plots a simple graph image.
- Saves PNG images to `map/plots/<mapname>.png`.

Usage:
    python scripts/plot_maps.py

Optional: pass a specific map folder path as an argument.
"""
import csv
import math
import os
import sys
from pathlib import Path
import argparse

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
# Default to the maps bundled in the environment folder
MAP_DIR = ROOT / "drp_env" / "map"
OUT_DIR = MAP_DIR / "plots"


def read_nodes(node_path):
    nodes = {}
    with open(node_path, newline='') as f:
        reader = csv.reader(f)
        headers = None
        # try to detect header or plain rows
        first = next(reader, None)
        if first is None:
            return nodes
        # decide whether first row is header by checking non-numeric
        try:
            [float(first[1]), float(first[2])]
            # first row looks numeric => treat as data
            row = first
            # handle rows with id,x,y or x,y
            if len(row) >= 3:
                nodes[int(row[0])] = (float(row[1]), float(row[2]))
            else:
                nodes[0] = (float(row[0]), float(row[1]))
        except Exception:
            # header present; assume columns include id,x,y names
            headers = [c.strip() for c in first]
        if headers is not None:
            for r in reader:
                if not r or all([c.strip()=='' for c in r]):
                    continue
                # try to find id,x,y in different orders
                vals = {h: v for h, v in zip(headers, r)}
                # common names
                id_keys = [k for k in headers if k.lower().replace('"','') in ('id','node','idx','index','node_id')]
                x_keys = [k for k in headers if k.lower().replace('"','') in ('x','posx','pos_x')]
                y_keys = [k for k in headers if k.lower().replace('"','') in ('y','posy','pos_y')]
                if id_keys and x_keys and y_keys:
                    nid = int(vals[id_keys[0]])
                    x = float(vals[x_keys[0]])
                    y = float(vals[y_keys[0]])
                    nodes[nid] = (x, y)
                else:
                    # fallback: try positional
                    try:
                        nid = int(r[0])
                        x = float(r[1])
                        y = float(r[2])
                        nodes[nid] = (x, y)
                    except Exception:
                        continue
        for r in reader:
            pass
    return nodes


def read_edges(edge_path):
    edges = []
    with open(edge_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all([c.strip()=='' for c in row]):
                continue
            try:
                a = int(row[0])
                b = int(row[1])
                edges.append((a, b))
            except Exception:
                # maybe header present, try to skip
                continue
    return edges


def plot_map(map_path, out_path):
    node_file = None
    edge_file = None
    # look for common filenames
    for name in ("node.csv", "_node.csv", "node.csv", "node.csv"):
        p = map_path / name
        if p.exists():
            node_file = p
            break
    for name in ("edge.csv",):
        p = map_path / name
        if p.exists():
            edge_file = p
            break
    if node_file is None or edge_file is None:
        print(f"Skipping {map_path.name}: node.csv or edge.csv not found")
        return False

    nodes = read_nodes(node_file)
    edges = read_edges(edge_file)

    if not nodes:
        print(f"No nodes parsed for {map_path.name}")
        return False

    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c='black')

    # annotate node ids
    for nid, (x, y) in nodes.items():
        plt.text(x, y, str(nid), fontsize=8, color='blue')

    # draw edges
    for a, b in edges:
        if a in nodes and b in nodes:
            x1, y1 = nodes[a]
            x2, y2 = nodes[b]
            plt.plot([x1, x2], [y1, y2], color='gray')

    plt.title(map_path.name)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")
    return True


def find_map_dirs(root):
    for child in root.iterdir():
        if child.is_dir():
            # skip plots folder
            if child.name == 'plots':
                continue
            yield child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mapdir', nargs='?', default=str(MAP_DIR), help='path to map directory (default: drp_env/map/)')
    parser.add_argument('--map', help='plot only this map subfolder name')
    args = parser.parse_args()

    map_root = Path(args.mapdir)
    out_root = map_root / 'plots'
    out_root.mkdir(parents=True, exist_ok=True)

    if args.map:
        mp = map_root / args.map
        if not mp.exists():
            print(f"Map {args.map} not found under {map_root}")
            sys.exit(1)
        plot_map(mp, out_root / f"{args.map}.png")
        return

    for mp in find_map_dirs(map_root):
        out_file = out_root / f"{mp.name}.png"
        plot_map(mp, out_file)


if __name__ == '__main__':
    main()
