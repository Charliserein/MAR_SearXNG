"""Bounded Serper Lens / ImgBB client. Never evaluate provider response text."""
import asyncio
import base64
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx


def valid_url(value):
    return isinstance(value, str) and urlparse(value).scheme in ("http", "https") and bool(urlparse(value).netloc)


class LensError(RuntimeError):
    """Contains a stage and error class, never API keys or response bodies."""


@dataclass
class SerperLensResult:
    title: str
    source: str
    link: str
    image_url: str
    thumbnail_url: str = ""

    @classmethod
    def from_dict(cls, data):
        return cls(title=data.get("title", ""), source=data.get("source", ""),
                   link=data.get("link", ""), image_url=data.get("imageUrl", ""),
                   thumbnail_url=data.get("thumbnailUrl", ""))


class SerperLensClient:
    def __init__(self, api_key: str, max_results: int = 20, timeout: float = 30, attempts: int = 2):
        if not api_key:
            raise LensError("MAR_SERPER_API_KEYS is missing")
        self.api_key, self.max_results = api_key, max_results
        self.timeout, self.attempts = timeout, attempts
        self.base_url = "https://google.serper.dev/lens"

    async def _post(self, url, stage, **kwargs):
        error = "unknown"
        async with httpx.AsyncClient(timeout=self.timeout,
                                     proxy=os.getenv("MAR_HTTP_PROXY") or None,
                                     trust_env=not bool(os.getenv("MAR_HTTP_PROXY"))) as client:
            for attempt in range(self.attempts):
                retry = True
                try:
                    response = await client.post(url, **kwargs)
                    response.raise_for_status()
                    result = response.json()
                    if not isinstance(result, dict):
                        raise ValueError("Expected JSON object")
                    return result
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    error = f"HTTP {status}"
                    # Store only a fixed diagnostic category, never provider text or credentials.
                    body = exc.response.text.lower()
                    if any(word in body for word in ("not enough credits", "insufficient credits",
                                                      "credits exhausted", "quota exceeded")):
                        error += " quota_exhausted"
                    retry = status == 429 or status >= 500
                except (httpx.HTTPError, ValueError) as exc:
                    error = type(exc).__name__
                    retry = isinstance(exc, httpx.HTTPError)
                if not retry or attempt + 1 == self.attempts:
                    break
                await asyncio.sleep(attempt + 1)
        raise LensError(f"{stage} failed: {error}") from None

    async def _upload_to_imgbb(self, image_path: Path, api_key: str, expiration: int = 600):
        if not api_key:
            raise LensError("MAR_IMGBB_API_KEYS is missing")
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        data = await self._post("https://api.imgbb.com/1/upload", "imgbb_upload",
                                data={"key": api_key, "image": encoded, "expiration": expiration})
        url = data.get("data", {}).get("url") if isinstance(data.get("data"), dict) else None
        if data.get("success") is not True or not valid_url(url):
            raise LensError("imgbb_upload returned an invalid response")
        return url.strip()

    async def search(self, image_path: Path, imgbb_api_key: str):
        public_url = await self._upload_to_imgbb(image_path, imgbb_api_key)
        return await self._search_by_url(public_url)

    async def _search_by_url(self, image_url: str):
        data = await self._post(self.base_url, "serper_lens",
                                headers={"X-API-KEY": self.api_key}, json={"url": image_url})
        organic = data.get("organic")
        if not isinstance(organic, list):
            raise LensError("serper_lens returned an invalid organic array")
        results = []
        for item in organic[:self.max_results]:
            if not isinstance(item, dict) or not valid_url(item.get("link")) or not valid_url(item.get("imageUrl")):
                raise LensError("serper_lens returned a result without valid page/image URLs")
            results.append(SerperLensResult.from_dict(item))
        return results
