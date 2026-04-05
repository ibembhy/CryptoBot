from __future__ import annotations

import base64
from pathlib import Path
import unittest

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_btc_bot.markets.auth import KalshiAuthConfig, KalshiAuthSigner


class KalshiAuthTests(unittest.TestCase):
    def test_websocket_headers_include_signed_auth_fields(self):
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        key_path = base_dir / "kalshi_test.key"
        key_path.write_bytes(private_key_bytes)

        signer = KalshiAuthSigner(
            KalshiAuthConfig(
                api_key_id="test-key-id",
                private_key_path=str(key_path),
            )
        )

        headers = signer.websocket_headers("/trade-api/ws/v2")
        self.assertEqual(headers["KALSHI-ACCESS-KEY"], "test-key-id")
        self.assertTrue(headers["KALSHI-ACCESS-TIMESTAMP"].isdigit())

        signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
        message = f"{headers['KALSHI-ACCESS-TIMESTAMP']}GET/trade-api/ws/v2".encode("utf-8")
        private_key.public_key().verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

    def test_request_headers_sign_trade_api_v2_path(self):
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        key_path = base_dir / "kalshi_test.key"
        key_path.write_bytes(private_key_bytes)

        signer = KalshiAuthSigner(
            KalshiAuthConfig(
                api_key_id="test-key-id",
                private_key_path=str(key_path),
            )
        )
        headers = signer.request_headers(method="POST", path="/portfolio/orders?limit=5")
        signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
        message = f"{headers['KALSHI-ACCESS-TIMESTAMP']}POST/trade-api/v2/portfolio/orders".encode("utf-8")
        private_key.public_key().verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )


if __name__ == "__main__":
    unittest.main()
