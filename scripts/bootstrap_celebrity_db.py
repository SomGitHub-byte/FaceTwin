#!/usr/bin/env python3
"""Generate celebrity DB with synthetic embeddings (no network, no DeepFace)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NAMES = [
    "Elon Musk", "Jeff Bezos", "Oprah Winfrey", "Tim Cook", "Satya Nadella",
    "Leonardo DiCaprio", "Meryl Streep", "Tom Hanks", "Zendaya", "Dwayne Johnson",
    "Beyoncé", "Taylor Swift", "Drake", "Rihanna", "Serena Williams",
    "LeBron James", "Lionel Messi", "Cristiano Ronaldo", "Stephen Curry",
    "Neil deGrasse Tyson", "Malala Yousafzai", "Barack Obama", "Michelle Obama",
    "Keanu Reeves", "Scarlett Johansson", "Ryan Gosling", "Emma Watson",
    "Chris Hemsworth", "Jennifer Lawrence", "Brad Pitt", "Angelina Jolie",
    "Will Smith", "Viola Davis", "Morgan Freeman", "Samuel L. Jackson",
    "Natalie Portman", "Robert Downey Jr.", "Lady Gaga", "Ed Sheeran",
    "Billie Eilish", "Gordon Ramsay", "MrBeast", "Mark Zuckerberg",
    "Sundar Pichai", "Jensen Huang", "Simone Biles", "Roger Federer",
    "Usain Bolt", "David Beckham", "Tom Cruise",
]

DIM = 512


def main() -> None:
    rng = np.random.default_rng(42)
    embeddings = []
    for name in NAMES:
        v = rng.standard_normal(DIM)
        v = v / np.linalg.norm(v)
        embeddings.append(v)

    emb_array = np.asarray(embeddings, dtype=np.float64)
    ref_scores = []
    for _ in range(200):
        q = rng.standard_normal(DIM)
        q = q / np.linalg.norm(q)
        ref_scores.extend((emb_array @ q).tolist())

    out = PROJECT_ROOT / "data" / "celebrity_embeddings.pkl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(
            {
                "names": NAMES,
                "embeddings": emb_array,
                "model": "Facenet512",
                "reference_scores": ref_scores,
                "score_stats": {
                    "min": float(min(ref_scores)),
                    "max": float(max(ref_scores)),
                    "mean": float(sum(ref_scores) / len(ref_scores)),
                },
            },
            f,
        )
    print(f"Wrote {len(NAMES)} synthetic embeddings to {out}")


if __name__ == "__main__":
    main()
