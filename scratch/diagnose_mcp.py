from __future__ import annotations

import asyncio
import base64
import io
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from agents.mcp_client import call_mcp_tool

SAMPLE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/7/7d/Dwayne_Johnson_2015.jpg",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
]


def fetch_sample_image() -> bytes:
    for url in SAMPLE_URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
                if data:
                    print("Downloaded sample image from", url)
                    return data
        except Exception as exc:
            print("Failed to download from", url, "->", exc)
    raise RuntimeError("Could not download a sample face image")


def main() -> None:
    print("Downloading sample face image...")
    image_bytes = fetch_sample_image()
    img = Image.open(io.BytesIO(image_bytes))
    print("Sample image format", img.format, "size", img.size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    async def run_tests() -> None:
        print("Calling moderate_image...")
        mod = await call_mcp_tool("moderate_image", {"image_b64": b64})
        print(mod)

        print("Calling detect_face...")
        face = await call_mcp_tool("detect_face", {"image_b64": b64})
        print(face)

        if face.get("ok"):
            print("Calling generate_embedding...")
            emb = await call_mcp_tool("generate_embedding", {"image_b64": b64})
            print(emb)
            if emb.get("ok"):
                print("Calling query_celebrity_db...")
                q = await call_mcp_tool("query_celebrity_db", {"vector": emb["vector"], "top_k": 5})
                print(q)

    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
