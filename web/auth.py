import hashlib
import hmac
import os
import re
from datetime import datetime

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ALIAS_RE = re.compile(r"^[\w' -]{2,30}$")   


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${key.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, key_hex = stored.split("$")
        key = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex),
                             n=2**14, r=8, p=1, dklen=32)
        return hmac.compare_digest(key, bytes.fromhex(key_hex))
    except ValueError:
        return False


def validate_registration(full_name: str, email: str, password: str,
                          password2: str, consent: bool) -> list[str]:
    errors = []
    if len(full_name.strip().split()) < 2 or len(full_name.strip()) > 100:
        errors.append("Escribe tu nombre completo (nombre y apellido).")
    if not EMAIL_RE.match(email.strip()):
        errors.append("El correo no es válido.")
    if not 8 <= len(password) <= 128:
        errors.append("La contraseña debe tener entre 8 y 128 caracteres.")
    if password != password2:
        errors.append("Las contraseñas no coinciden.")
    if not consent:
        errors.append("Debes aceptar el tratamiento de datos para crear la cuenta.")
    return errors


def validate_profile(alias: str, birth_year: str) -> tuple[list[str], int | None]:
    errors, year = [], None
    if not ALIAS_RE.match(alias.strip()):
        errors.append("El apodo debe tener entre 2 y 30 caracteres (letras, números, espacios o guiones).")
    this_year = datetime.now().year
    try:
        year = int(birth_year)
        if not this_year - 100 <= year <= this_year:
            raise ValueError
    except ValueError:
        year = None
        errors.append("Escribe un año de nacimiento válido.")
    return errors, year