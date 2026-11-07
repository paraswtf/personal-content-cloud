"""The single owner account (docs/ARCHITECTURE.md section 4).

Contract:
- get_account(conn) -> sqlite3.Row | None: the row with id 1.
- create_account(conn, username, password) -> sqlite3.Row: only while no account exists;
  raises AccountExists otherwise (also when two setups race).
- check_credentials(conn, username, password) -> bool: case-insensitive user name, argon2 verify,
  rehash on success when the parameters changed; constant work for unknown names.
- check_password(conn, password) -> bool: the password alone (vault unlock, password change).
- change_password(conn, new_password): stores a new hash.
- username_key(name): the casefolded form stored in account.username_key.
"""
from __future__ import annotations

import logging
import sqlite3

from app.auth import passwords
from app.db import tx
from app.util import clean_name, now_iso

log = logging.getLogger(__name__)

ACCOUNT_ID = 1


class AccountExists(Exception):
    """setup was called while an account already exists."""


def username_key(name: str) -> str:
    return clean_name(name, 200).casefold()


def get_account(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM account WHERE id = ?", (ACCOUNT_ID,)).fetchone()


def create_account(conn: sqlite3.Connection, username: str, password: str) -> sqlite3.Row:
    name = clean_name(username, 64)
    if not name:
        raise ValueError("username is empty")
    if get_account(conn) is not None:
        raise AccountExists()
    hashed = passwords.hash_password(password)  # slow: outside the write transaction
    now = now_iso()
    try:
        with tx(conn):
            if get_account(conn) is not None:
                raise AccountExists()
            conn.execute(
                "INSERT INTO account(id, username, username_key, password_hash, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ACCOUNT_ID, name, username_key(name), hashed, now, now),
            )
    except sqlite3.IntegrityError as exc:
        raise AccountExists() from exc
    log.info("account created")
    row = get_account(conn)
    assert row is not None
    return row


def _verify_and_upgrade(conn: sqlite3.Connection, row: sqlite3.Row, password: str) -> bool:
    hashed = row["password_hash"]
    if not passwords.verify_password(hashed, password):
        return False
    if passwords.needs_rehash(hashed):
        new_hash = passwords.hash_password(password)
        with tx(conn):
            conn.execute(
                "UPDATE account SET password_hash = ?, updated_at = ? WHERE id = ? AND password_hash = ?",
                (new_hash, now_iso(), ACCOUNT_ID, hashed),
            )
        log.info("password hash upgraded to current parameters")
    return True


def check_credentials(conn: sqlite3.Connection, username: str, password: str) -> bool:
    row = get_account(conn)
    if row is None or row["username_key"] != username_key(username or ""):
        passwords.dummy_verify(password or "")
        return False
    return _verify_and_upgrade(conn, row, password or "")


def check_password(conn: sqlite3.Connection, password: str) -> bool:
    row = get_account(conn)
    if row is None:
        passwords.dummy_verify(password or "")
        return False
    return _verify_and_upgrade(conn, row, password or "")


def change_password(conn: sqlite3.Connection, new_password: str) -> None:
    hashed = passwords.hash_password(new_password)
    with tx(conn):
        cur = conn.execute(
            "UPDATE account SET password_hash = ?, updated_at = ? WHERE id = ?", (hashed, now_iso(), ACCOUNT_ID)
        )
        if cur.rowcount != 1:
            raise LookupError("no account")
    log.info("password changed")
