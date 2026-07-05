#!/usr/bin/env python3
"""Build offline celebrity embedding database (run once, commit the .pkl)."""

from __future__ import annotations

import io
import logging
import os
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Curated ~50 public figures across categories (Wikipedia portrait URLs)
CELEBRITIES: list[tuple[str, str]] = [
    ("Elon Musk", "https://upload.wikimedia.org/wikipedia/commons/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg"),
    ("Jeff Bezos", "https://upload.wikimedia.org/wikipedia/commons/0/03/Jeff_Bezos_at_Amazon_Spheres_Grand_Opening_in_2018.jpg"),
    ("Oprah Winfrey", "https://upload.wikimedia.org/wikipedia/commons/8/8e/Oprah_in_2014.jpg"),
    ("Tim Cook", "https://upload.wikimedia.org/wikipedia/commons/7/7d/Tim_Cook_2009_cropped.jpg"),
    ("Satya Nadella", "https://upload.wikimedia.org/wikipedia/commons/7/7c/Satya_Nadella.jpg"),
    ("Leonardo DiCaprio", "https://upload.wikimedia.org/wikipedia/commons/4/46/Leonardo_Dicaprio_Cannes_2019.jpg"),
    ("Meryl Streep", "https://upload.wikimedia.org/wikipedia/commons/4/46/Meryl_Streep_December_2014.jpg"),
    ("Tom Hanks", "https://upload.wikimedia.org/wikipedia/commons/a/a9/Tom_Hanks_TIFF_2019.jpg"),
    ("Zendaya", "https://upload.wikimedia.org/wikipedia/commons/2/28/Zendaya_-_2019_by_Glenn_Francis.jpg"),
    ("Dwayne Johnson", "https://upload.wikimedia.org/wikipedia/commons/1/1f/Dwayne_Johnson_2014.jpg"),
    ("Beyoncé", "https://upload.wikimedia.org/wikipedia/commons/1/17/Beyonce_at_The_Lion_King_European_Premiere_2019.png"),
    ("Taylor Swift", "https://upload.wikimedia.org/wikipedia/commons/b/b5/191125_Taylor_Swift_at_the_2019_American_Music_Awards_%28cropped%29.png"),
    ("Drake", "https://upload.wikimedia.org/wikipedia/commons/2/28/Drake_July_2016.jpg"),
    ("Rihanna", "https://upload.wikimedia.org/wikipedia/commons/1/16/Rihanna_Fenty_2019.png"),
    ("Serena Williams", "https://upload.wikimedia.org/wikipedia/commons/4/4b/Serena_Williams_at_2013_US_Open.jpg"),
    ("LeBron James", "https://upload.wikimedia.org/wikipedia/commons/7/7a/LeBron_James_Lakers.jpg"),
    ("Lionel Messi", "https://upload.wikimedia.org/wikipedia/commons/b/b4/Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg"),
    ("Cristiano Ronaldo", "https://upload.wikimedia.org/wikipedia/commons/8/8c/Cristiano_Ronaldo_2018.jpg"),
    ("Stephen Curry", "https://upload.wikimedia.org/wikipedia/commons/3/36/Stephen_Curry_dribbling_2016_%28cropped%29.jpg"),
    ("Neil deGrasse Tyson", "https://upload.wikimedia.org/wikipedia/commons/1/1e/Neil_deGrasse_Tyson_at_2017_Starmus_Festival_%28cropped%29.jpg"),
    ("Malala Yousafzai", "https://upload.wikimedia.org/wikipedia/commons/0/03/Malala_Yousafzai_2015.jpg"),
    ("Barack Obama", "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"),
    ("Michelle Obama", "https://upload.wikimedia.org/wikipedia/commons/4/4b/Michelle_Obama_2013_official_portrait.jpg"),
    ("Keanu Reeves", "https://upload.wikimedia.org/wikipedia/commons/9/90/Keanu_Reeves_%28crop_and_levels%29.jpg"),
    ("Scarlett Johansson", "https://upload.wikimedia.org/wikipedia/commons/2/2a/Scarlett_Johansson-8585.jpg"),
    ("Ryan Gosling", "https://upload.wikimedia.org/wikipedia/commons/e/e5/Ryan_Gosling_2018_%28cropped%29.jpg"),
    ("Emma Watson", "https://upload.wikimedia.org/wikipedia/commons/7/7f/Emma_Watson_2013.jpg"),
    ("Chris Hemsworth", "https://upload.wikimedia.org/wikipedia/commons/e/e8/Chris_Hemsworth_by_Gage_Skidmore_2_%28cropped%29.jpg"),
    ("Jennifer Lawrence", "https://upload.wikimedia.org/wikipedia/commons/0/0b/Jennifer_Lawrence_SDCC_2015_X-Men.jpg"),
    ("Brad Pitt", "https://upload.wikimedia.org/wikipedia/commons/4/4e/Brad_Pitt_2019_by_Glenn_Francis.jpg"),
    ("Angelina Jolie", "https://upload.wikimedia.org/wikipedia/commons/1/12/Angelina_Jolie_2_July_2020_%28cropped%29.jpg"),
    ("Will Smith", "https://upload.wikimedia.org/wikipedia/commons/1/15/Will_Smith_at_the_2024_Toronto_International_Film_Festival_%28cropped%29.jpg"),
    ("Viola Davis", "https://upload.wikimedia.org/wikipedia/commons/9/9d/Viola_Davis_by_Gage_Skidmore.jpg"),
    ("Morgan Freeman", "https://upload.wikimedia.org/wikipedia/commons/4/4f/Morgan_Freeman_2010.jpg"),
    ("Samuel L. Jackson", "https://upload.wikimedia.org/wikipedia/commons/2/29/SamuelLJackson.jpg"),
    ("Natalie Portman", "https://upload.wikimedia.org/wikipedia/commons/7/7e/Natalie_Portman_Cannes_2015_5_%28cropped%29.jpg"),
    ("Robert Downey Jr.", "https://upload.wikimedia.org/wikipedia/commons/2/2e/Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg"),
    ("Lady Gaga", "https://upload.wikimedia.org/wikipedia/commons/0/0e/Lady_Gaga_at_Joanne_World_Tour_%28cropped%29.jpg"),
    ("Ed Sheeran", "https://upload.wikimedia.org/wikipedia/commons/c/c1/Ed_Sheeran-6886_%28cropped%29.jpg"),
    ("Billie Eilish", "https://upload.wikimedia.org/wikipedia/commons/9/99/Billie_Eilish_2019_by_Glenn_Francis_%28cropped%29.jpg"),
    ("Gordon Ramsay", "https://upload.wikimedia.org/wikipedia/commons/6/6f/Gordon_Ramsay.jpg"),
    ("MrBeast", "https://upload.wikimedia.org/wikipedia/commons/c/ce/MrBeast_%282%29_%28cropped%29.jpg"),
    ("Mark Zuckerberg", "https://upload.wikimedia.org/wikipedia/commons/1/18/Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg"),
    ("Sundar Pichai", "https://upload.wikimedia.org/wikipedia/commons/4/4e/Sundar_pichai.png"),
    ("Jensen Huang", "https://upload.wikimedia.org/wikipedia/commons/0/0c/Jensen_Huang_%28cropped%29.jpg"),
    ("Simone Biles", "https://upload.wikimedia.org/wikipedia/commons/4/4d/Simone_Biles%2C_June_2021_%28cropped%29.jpg"),
    ("Roger Federer", "https://upload.wikimedia.org/wikipedia/commons/3/3e/Roger_Federer_%282015%29_%28cropped%29.jpg"),
    ("Usain Bolt", "https://upload.wikimedia.org/wikipedia/commons/1/15/Usain_Bolt_after_200m_qualifications_Olympic_Games_2012.jpg"),
    ("David Beckham", "https://upload.wikimedia.org/wikipedia/commons/7/79/David_Beckham_2010.jpg"),
]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _download_image(url: str) -> np.ndarray | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FaceTwin/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)[:, :, ::-1]  # BGR
    except Exception as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        return None


def _embed_image(bgr: np.ndarray, model: str) -> list[float] | None:
    from deepface import DeepFace

    try:
        result = DeepFace.represent(
            img_path=bgr,
            model_name=model,
            enforce_detection=True,
            detector_backend="opencv",
        )
        return result[0]["embedding"]
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return None


def build(model: str = "Facenet512", synthetic_fallback: bool = True) -> Path:
    from config import EMBEDDING_MODEL

    model = model or EMBEDDING_MODEL
    names: list[str] = []
    embeddings: list[list[float]] = []
    seen: set[str] = set()

    for name, url in CELEBRITIES:
        if name in seen:
            continue
        seen.add(name)
        logger.info("Processing %s...", name)
        bgr = _download_image(url)
        if bgr is not None:
            vec = _embed_image(bgr, model)
            if vec:
                names.append(name)
                embeddings.append(vec)
                continue
        if synthetic_fallback:
            rng = np.random.default_rng(abs(hash(name)) % (2**32))
            vec = rng.standard_normal(512).tolist()
            norm = np.linalg.norm(vec)
            vec = (np.array(vec) / norm).tolist()
            names.append(name)
            embeddings.append(vec)
            logger.info("  Used synthetic embedding for %s", name)

    if not names:
        raise RuntimeError("No celebrity embeddings generated")

    emb_array = np.asarray(embeddings, dtype=np.float64)
    # Precompute reference score distribution (random query simulation)
    rng = np.random.default_rng(42)
    ref_scores = []
    for _ in range(200):
        q = rng.standard_normal(emb_array.shape[1])
        q = q / np.linalg.norm(q)
        sims = emb_array @ q
        ref_scores.extend(sims.tolist())

    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "celebrity_embeddings.pkl"

    payload = {
        "names": names,
        "embeddings": emb_array,
        "model": model,
        "reference_scores": ref_scores,
        "score_stats": {
            "min": float(np.min(ref_scores)),
            "max": float(np.max(ref_scores)),
            "mean": float(np.mean(ref_scores)),
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info("Saved %d celebrities to %s", len(names), out_path)
    return out_path


if __name__ == "__main__":
    build()
