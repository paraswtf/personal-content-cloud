"""Seed for the vault and trash screens (ios/UITests/VaultTrashScreens.swift); see ci/seed_server.py.

library: three drawn photos in the trash, named after the area: the test restores the first, deletes the second
for good and only previews the third. They are dated February 2026 (older than the core's items) and deleted a day
before the core's trashed photo, which the tests never touch.

api: opens the vault for the app's device token. The app signs in with a session of its own, so the vault the
core's seed opens for the owner's browser session stays locked for the app; this opens it there too, for the
screens that need it open (Settings adding a device, for example). It stays open for VAULT_UNLOCK_MIN minutes
(15 unless the server is told otherwise); the vault test locks it and opens it again with the password.
"""
from datetime import timedelta

TRASHED = ("VAULTTRASH_RESTORE.JPG", "VAULTTRASH_PURGE.JPG", "VAULTTRASH_KEEP.JPG")


def library(kit):
    deleted = (kit.end - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    uids = {}
    for k, name in enumerate(TRASHED):
        img = kit.scene(7100 + k, [])
        item = kit.make_item(kit.conn, data=kit.jpeg(img), width=kit.W, height=kit.H, filename=name,
                             taken_at=f"2026-02-{10 + k:02d}T09:30:00.000Z", camera="CI Camera", category="misc")
        kit.previews(item, img)
        with kit.tx(kit.conn):
            kit.conn.execute("UPDATE items SET deleted_at = ? WHERE id = ?", (deleted, item["id"]))
        uids[name] = item["uid"]
    return {"trashed": uids}


def api(kit):
    bearer = {"Authorization": f"Bearer {kit.token}"}
    state = kit.Client(kit.server).call("POST", "/api/auth/vault/unlock", {"password": kit.password}, headers=bearer)
    return {"app_vault_until": state.get("vault_until")}
