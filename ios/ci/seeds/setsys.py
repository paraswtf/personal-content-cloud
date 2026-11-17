"""Seed of the system side of Settings (ios/UITests/SettingsSystemScreens.swift).

The System and Learning screens read the server as it is (models off, no GPU on the CI machine, backups on),
so they need nothing here. The undo history test needs a change the app makes on data of its own: the album
"Setsys undo test", whose description the app's Debug-only test hook changes through its undo history
(Features/Settings/System/SettingsSystemTestHook.swift), then undoes and redoes. Nothing else is touched.
"""

ALBUM = "Setsys undo test"


def api(kit):
    album = kit.client.call("POST", "/api/albums", {"name": ALBUM, "description": "Made by the setsys seed"})
    return {"album_id": album["id"], "album": ALBUM}
