from __future__ import annotations

import base64
import ctypes
import os
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_ubyte,
    c_void_p,
    cast,
    create_string_buffer,
)
from ctypes.wintypes import BOOL, DWORD


class CredentialStoreError(RuntimeError):
    pass


class _DataBlob(Structure):
    _fields_ = [("cbData", DWORD), ("pbData", POINTER(c_ubyte))]


_ENTROPY = b"kellyClipAi:llm-api-key:v1"
_OAUTH_ENTROPY = b"kellyClipAi:oauth-token:v1"
_OAUTH_TOKEN_PREFIX = "dpapi:v1:"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


def _windows_api():
    if os.name != "nt":
        raise CredentialStoreError("El almacenamiento cifrado requiere Windows.")
    crypt32 = ctypes.WinDLL("Crypt32.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
    return crypt32, kernel32


def _blob(data: bytes):
    buffer = create_string_buffer(data)
    return _DataBlob(len(data), cast(buffer, POINTER(c_ubyte))), buffer


def _protect(data: bytes, decrypt: bool, entropy: bytes = _ENTROPY) -> bytes:
    crypt32, kernel32 = _windows_api()
    input_blob, input_buffer = _blob(data)
    entropy_blob, entropy_buffer = _blob(entropy)
    output_blob = _DataBlob()
    function = (
        crypt32.CryptUnprotectData
        if decrypt
        else crypt32.CryptProtectData
    )
    if decrypt:
        function.argtypes = [
            POINTER(_DataBlob),
            c_void_p,
            POINTER(_DataBlob),
            c_void_p,
            c_void_p,
            DWORD,
            POINTER(_DataBlob),
        ]
        function.restype = BOOL
        succeeded = function(
            byref(input_blob),
            None,
            byref(entropy_blob),
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            byref(output_blob),
        )
    else:
        function.argtypes = [
            POINTER(_DataBlob),
            ctypes.c_wchar_p,
            POINTER(_DataBlob),
            c_void_p,
            c_void_p,
            DWORD,
            POINTER(_DataBlob),
        ]
        function.restype = BOOL
        succeeded = function(
            byref(input_blob),
            None,
            byref(entropy_blob),
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            byref(output_blob),
        )
    if not succeeded:
        raise CredentialStoreError(
            "Windows no pudo proteger o recuperar la clave. Inicia sesión en la misma cuenta de Windows."
        )
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree.argtypes = [c_void_p]
        kernel32.LocalFree.restype = c_void_p
        kernel32.LocalFree(output_blob.pbData)
        del input_buffer, entropy_buffer


def encrypt_api_key(api_key: str) -> bytes:
    if not api_key.strip():
        raise CredentialStoreError("La clave API no puede estar vacía.")
    return _protect(api_key.encode("utf-8"), decrypt=False)


def decrypt_api_key(ciphertext: bytes) -> str:
    try:
        return _protect(ciphertext, decrypt=True).decode("utf-8")
    except UnicodeDecodeError as error:
        raise CredentialStoreError(
            "La clave guardada no se pudo recuperar. Vuelve a introducirla."
        ) from error


def is_oauth_token_encrypted(token: str) -> bool:
    return token.startswith(_OAUTH_TOKEN_PREFIX)


def encrypt_oauth_token(token: str) -> str:
    if not token:
        raise CredentialStoreError("El token OAuth no puede estar vacío.")
    if is_oauth_token_encrypted(token):
        return token
    ciphertext = _protect(token.encode("utf-8"), decrypt=False, entropy=_OAUTH_ENTROPY)
    encoded = base64.urlsafe_b64encode(ciphertext).decode("ascii")
    return f"{_OAUTH_TOKEN_PREFIX}{encoded}"


def decrypt_oauth_token(token: str | None) -> str | None:
    if token is None:
        return None
    if not is_oauth_token_encrypted(token):
        raise CredentialStoreError(
            "Se encontró un token OAuth sin cifrar. Reinicia la aplicación para migrarlo."
        )
    encoded = token.removeprefix(_OAUTH_TOKEN_PREFIX)
    try:
        ciphertext = base64.b64decode(encoded, altchars=b"-_", validate=True)
        return _protect(ciphertext, decrypt=True, entropy=_OAUTH_ENTROPY).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise CredentialStoreError(
            "El token OAuth guardado no se pudo recuperar. Vuelve a conectar la cuenta."
        ) from error
