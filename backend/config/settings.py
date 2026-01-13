"""
Application Settings - Environment-based configuration.

Unlike constants.py, these settings CAN be configured via environment variables.
However, they must NOT override the hard-coded risk constants.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # Environment Mode
    # ═══════════════════════════════════════════════════════════════════
    binance_mode: Literal["testnet", "live"] = Field(
        default="testnet",
        description="Trading environment: testnet or live"
    )
    debug: bool = Field(default=False, description="Enable debug logging")
    
    # ═══════════════════════════════════════════════════════════════════
    # Database
    # ═══════════════════════════════════════════════════════════════════
    database_url: str = Field(
        default="sqlite+aiosqlite:///./trading.db",
        description="Database connection string (SQLite for dev, PostgreSQL for prod)"
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # API Server
    # ═══════════════════════════════════════════════════════════════════
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8001, description="API server port")
    
    # ═══════════════════════════════════════════════════════════════════
    # Binance API (Keys are encrypted at rest via KeyVault)
    # ═══════════════════════════════════════════════════════════════════
    binance_api_key_encrypted: str = Field(
        default="",
        description="Encrypted Binance API key"
    )
    binance_api_secret_encrypted: str = Field(
        default="",
        description="Encrypted Binance API secret"
    )
    
    # Encryption key for API credentials (must be 32 bytes base64)
    encryption_key: str = Field(
        default="",
        description="Fernet encryption key for API credentials"
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # Symbol Selection
    # ═══════════════════════════════════════════════════════════════════
    symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        description="Trading symbols (max 3)"
    )
    auto_select_symbols: bool = Field(
        default=False,
        description="Auto-select top 3 by 24h volume"
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # Paths
    # ═══════════════════════════════════════════════════════════════════
    @property
    def project_root(self) -> Path:
        """Return the project root directory."""
        return Path(__file__).parent.parent
    
    @property
    def logs_dir(self) -> Path:
        """Return the logs directory, creating if needed."""
        logs = self.project_root / "logs"
        logs.mkdir(exist_ok=True)
        return logs
    
    # ═══════════════════════════════════════════════════════════════════
    # Validators
    # ═══════════════════════════════════════════════════════════════════
    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v):
        """Parse symbols from comma-separated string or list."""
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return [s.upper() for s in v]
    
    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v):
        """Ensure max 3 symbols."""
        if len(v) > 3:
            raise ValueError("Maximum 3 concurrent symbols allowed")
        return v
    
    # ═══════════════════════════════════════════════════════════════════
    # Binance URLs
    # ═══════════════════════════════════════════════════════════════════
    @property
    def binance_base_url(self) -> str:
        """Return Binance Futures API base URL."""
        if self.binance_mode == "testnet":
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"
    
    @property
    def binance_ws_url(self) -> str:
        """Return Binance Futures WebSocket URL."""
        if self.binance_mode == "testnet":
            return "wss://stream.binancefuture.com"
        return "wss://fstream.binance.com"
    
    @property
    def is_testnet(self) -> bool:
        """Check if running in testnet mode."""
        return self.binance_mode == "testnet"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
