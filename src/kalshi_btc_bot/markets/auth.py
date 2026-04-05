from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


@dataclass(frozen=True)
class KalshiAuthConfig:
    api_key_id: str
    private_key_path: str


class KalshiAuthSigner:
    def __init__(self, config: KalshiAuthConfig) -> None:
        self.config = config
        self._private_key = self._load_private_key(config.private_key_path)

    def websocket_headers(self, path: str = "/trade-api/ws/v2") -> dict[str, str]:
        timestamp = self._timestamp_ms()
        signature = self._sign(timestamp=timestamp, method="GET", path=path)
        return {
            "KALSHI-ACCESS-KEY": self.config.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def request_headers(self, *, method: str, path: str) -> dict[str, str]:
        normalized_path = path.split("?", 1)[0]
        if not normalized_path.startswith("/trade-api/"):
            normalized_path = f"/trade-api/v2{normalized_path if normalized_path.startswith('/') else f'/{normalized_path}'}"
        timestamp = self._timestamp_ms()
        signature = self._sign(timestamp=timestamp, method=method, path=normalized_path)
        return {
            "KALSHI-ACCESS-KEY": self.config.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    @staticmethod
    def _load_private_key(path: str):
        private_key_bytes = Path(path).expanduser().read_bytes()
        return serialization.load_pem_private_key(private_key_bytes, password=None)

    def _sign(self, *, timestamp: str, method: str, path: str) -> str:
        message = f"{timestamp}{method.upper()}{path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    @staticmethod
    def _timestamp_ms() -> str:
        return str(int(datetime.now(timezone.utc).timestamp() * 1000))
