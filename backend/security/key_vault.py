"""
Encrypted API Key Vault

Securely stores and retrieves Binance API credentials using Fernet (AES-256) encryption.
Keys are encrypted at rest and only decrypted in memory during runtime.

Usage:
    # Generate a new encryption key (do this once, store securely)
    $ python -c "from security.key_vault import KeyVault; print(KeyVault.generate_key())"
    
    # Encrypt your API keys
    $ python -c "from security.key_vault import KeyVault; v = KeyVault('YOUR_ENCRYPTION_KEY'); print(v.encrypt('YOUR_API_KEY'))"
"""
import base64
import logging
from dataclasses import dataclass
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


@dataclass
class APICredentials:
    """Decrypted API credentials (held only in memory)."""
    api_key: str
    api_secret: str


class KeyVaultError(Exception):
    """Base exception for KeyVault errors."""
    pass


class KeyVault:
    """
    Secure storage and retrieval of encrypted API credentials.
    
    Uses Fernet symmetric encryption (AES-256 in CBC mode with HMAC).
    """
    
    def __init__(self, encryption_key: str):
        """
        Initialize the KeyVault with an encryption key.
        
        Args:
            encryption_key: Base64-encoded 32-byte Fernet key
        """
        if not encryption_key:
            raise KeyVaultError("Encryption key is required")
        
        try:
            self._fernet = Fernet(encryption_key.encode())
        except Exception as e:
            raise KeyVaultError(f"Invalid encryption key format: {e}")
    
    @staticmethod
    def generate_key() -> str:
        """
        Generate a new Fernet encryption key.
        
        Returns:
            Base64-encoded 32-byte key string
        """
        return Fernet.generate_key().decode()
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string.
        
        Args:
            plaintext: The string to encrypt
            
        Returns:
            Base64-encoded encrypted string
        """
        if not plaintext:
            raise KeyVaultError("Cannot encrypt empty string")
        
        encrypted = self._fernet.encrypt(plaintext.encode())
        return encrypted.decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt an encrypted string.
        
        Args:
            ciphertext: Base64-encoded encrypted string
            
        Returns:
            Decrypted plaintext string
        """
        if not ciphertext:
            raise KeyVaultError("Cannot decrypt empty string")
        
        try:
            decrypted = self._fernet.decrypt(ciphertext.encode())
            return decrypted.decode()
        except InvalidToken:
            raise KeyVaultError("Invalid ciphertext or wrong encryption key")
    
    def get_credentials(
        self,
        encrypted_api_key: str,
        encrypted_api_secret: str
    ) -> APICredentials:
        """
        Decrypt and return API credentials.
        
        Args:
            encrypted_api_key: Encrypted API key
            encrypted_api_secret: Encrypted API secret
            
        Returns:
            APICredentials with decrypted values
        """
        try:
            api_key = self.decrypt(encrypted_api_key)
            api_secret = self.decrypt(encrypted_api_secret)
            
            logger.info("API credentials decrypted successfully")
            return APICredentials(api_key=api_key, api_secret=api_secret)
            
        except KeyVaultError:
            logger.error("Failed to decrypt API credentials")
            raise


def setup_credentials_interactive() -> None:
    """
    Interactive CLI helper to encrypt API credentials.
    
    Run this once to generate encrypted credentials for your .env file.
    """
    import getpass
    
    print("\n" + "═" * 60)
    print("  INSTITUTIONAL TRADING PLATFORM - CREDENTIAL SETUP")
    print("═" * 60)
    
    # Step 1: Generate or use existing key
    print("\n[1/3] Encryption Key Setup")
    choice = input("Generate new encryption key? (y/n): ").strip().lower()
    
    if choice == 'y':
        encryption_key = KeyVault.generate_key()
        print(f"\n✓ New encryption key (save this securely!):")
        print(f"  ENCRYPTION_KEY={encryption_key}")
    else:
        encryption_key = getpass.getpass("Enter your existing encryption key: ")
    
    # Initialize vault
    try:
        vault = KeyVault(encryption_key)
    except KeyVaultError as e:
        print(f"\n✗ Error: {e}")
        return
    
    # Step 2: Encrypt API Key
    print("\n[2/3] Binance API Key")
    api_key = getpass.getpass("Enter your Binance API Key: ")
    encrypted_key = vault.encrypt(api_key)
    print(f"\n✓ Encrypted API Key:")
    print(f"  BINANCE_API_KEY_ENCRYPTED={encrypted_key}")
    
    # Step 3: Encrypt API Secret
    print("\n[3/3] Binance API Secret")
    api_secret = getpass.getpass("Enter your Binance API Secret: ")
    encrypted_secret = vault.encrypt(api_secret)
    print(f"\n✓ Encrypted API Secret:")
    print(f"  BINANCE_API_SECRET_ENCRYPTED={encrypted_secret}")
    
    # Summary
    print("\n" + "═" * 60)
    print("  SETUP COMPLETE - Add these to your .env file")
    print("═" * 60)
    print(f"\nENCRYPTION_KEY={encryption_key}")
    print(f"BINANCE_API_KEY_ENCRYPTED={encrypted_key}")
    print(f"BINANCE_API_SECRET_ENCRYPTED={encrypted_secret}")
    print("\n⚠️  NEVER commit your .env file to version control!")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    setup_credentials_interactive()
