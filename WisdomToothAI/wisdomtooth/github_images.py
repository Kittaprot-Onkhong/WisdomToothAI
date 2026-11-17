"""Download test images straight from a GitHub repository (Contents API)."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional

from .config import IMAGE_EXTENSIONS


def _open(url: str, token: Optional[str], retries: int = 3):
    headers = {"User-Agent": "WisdomToothAI", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):  # retrying will not help
                raise
            last_error = error
        except urllib.error.URLError as error:
            last_error = error
        time.sleep(attempt)
    raise last_error


def list_github_images(owner: str, repo: str, path: str = "test_images",
                       branch: str = "main", token: Optional[str] = None) -> List[dict]:
    """Return [{'name', 'size', 'download_url'}] for image files in repo/path."""
    if not owner or not repo:
        raise ValueError(
            "GitHub owner/repo is empty. Fill GITHUB_OWNER and GITHUB_REPO "
            "(Colab form or wisdomtooth/config.py)."
        )
    api = (f"https://api.github.com/repos/{owner}/{repo}/contents/"
           f"{urllib.parse.quote(path.strip('/'))}?ref={urllib.parse.quote(branch)}")
    try:
        with _open(api, token) as response:
            items = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        hint = {
            404: "not found — check owner/repo/branch/folder; if the repo is private, provide a GitHub token",
            401: "bad GitHub token",
            403: "forbidden or API rate limit reached — provide a GitHub token",
        }.get(error.code, "")
        raise RuntimeError(f"GitHub API error {error.code} for {api}: {hint}") from error

    if not isinstance(items, list):
        raise RuntimeError(f"'{path}' in {owner}/{repo} is not a folder.")

    images = [
        {"name": item["name"], "size": item.get("size"), "download_url": item.get("download_url")}
        for item in items
        if item.get("type") == "file"
        and Path(item["name"]).suffix.lower() in IMAGE_EXTENSIONS
        and item.get("download_url")
    ]
    return sorted(images, key=lambda item: item["name"].lower())


def download_github_images(owner: str, repo: str, dest: Path, path: str = "test_images",
                           branch: str = "main", token: Optional[str] = None,
                           limit: int = 0) -> List[Path]:
    """Download images into `dest` (skips files already present with same size).

    limit=0 downloads everything; otherwise only the first `limit` by filename.
    Returns the local paths, sorted by filename.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    images = list_github_images(owner, repo, path, branch, token)
    if not images:
        raise FileNotFoundError(
            f"No images found in {owner}/{repo}/{path} (branch '{branch}'). "
            "Upload the test images to that folder first."
        )
    total_available = len(images)
    if limit and limit > 0:
        images = images[:limit]

    local_paths = []
    for number, item in enumerate(images, start=1):
        target = dest / item["name"]
        if not (target.exists() and item["size"] is not None and target.stat().st_size == item["size"]):
            with _open(item["download_url"], token) as response:
                target.write_bytes(response.read())
        local_paths.append(target)
        if number % 25 == 0 or number == len(images):
            print(f"  downloaded {number}/{len(images)}")

    print(f"GitHub folder has {total_available} image(s); using {len(local_paths)}.")
    return local_paths
