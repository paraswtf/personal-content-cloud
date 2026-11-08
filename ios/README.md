# Content Cloud for iPhone

A small SwiftUI app that keeps your phone's new photos and videos flowing into your Personal Content
Cloud, and that can clear the ones it has synced out of the phone's library when you ask.

What it does:

- Signs in with a device token (Settings > Sessions > Devices on the website creates one; it is shown
  once). No password is stored on the phone.
- Watches the photo library and uploads what is new: photos, videos, and both halves of a Live Photo,
  as the original files, through the server's resumable upload protocol. Nothing is re-sent after a
  reinstall: the app asks the server what it already holds, by asset and by file hash.
- Uploads run in a background URLSession (`in.rivant.pcc.upload`), so a file keeps going while the
  app is in the background, and an upload cut off by a lost connection or a killed app continues
  from where the server's copy ends instead of starting again.
- Syncs while the app is open, when the library changes, and in the background when iOS gives it a
  processing window (usually when the phone is charging and idle). Options: Wi-Fi only, charging only,
  include videos.
- "Delete synced": lists the assets the server confirms it holds and that are not in the server's
  trash (it checks again right before), and removes them from the library through the system's own
  confirmation dialog. Assets whose server copy is in the trash are listed under their own heading
  and are never selected by "All". Nothing is deleted on its own.

## How uploads resume (docs/ARCHITECTURE.md section 28)

- Each asset's originals are exported into `Application Support/Uploads/assets/<asset>/` (excluded
  from backups) and hashed. The upload state (upload id, which half, next offset, the chunk file in
  flight, attempts) is stored in the asset's `SyncRecord`, so it survives the app being killed.
- Every request of the upload protocol (create, each chunk, complete) is an upload task of the
  background session with its body in a file. The delegate (`BackgroundUploader`) schedules the
  next chunk when one finishes. iOS relaunches the app into the delegate when tasks finish while it
  is not running (`handleEventsForBackgroundURLSession` in `AppDelegate`).
- The create sends the asset's local identifier as `pair_key` and `resource` (`still` or
  `video`). The server answers a repeated create from the same device with the upload it already
  has and its `received_bytes`, so a lost reply or a relaunch never opens a second upload.
- On every run the engine first asks `POST /api/uploads/check`. Its `pending` list gives the
  server's offset for this device's unfinished uploads. An upload it does not list had its last
  reply lost, and `GET /api/uploads/{id}` says what happened to it. Then the engine hashes and
  creates the rest. For assets the app has no record of (after a reinstall, or photos first
  uploaded from the web), the check also sends the file hashes, so nothing is sent twice and
  nothing the owner deleted is restored from the trash.
- Failed requests are retried after 1, 2, 4, 8, 16 and 30 seconds, eight attempts at most. A 409
  `offset_mismatch` continues from the server's offset. A 413 (file too large) or 507 (server out
  of space) fails the asset with the server's message, shown under Failed on the Sync screen.
- Wi-Fi only sets `allowsCellularAccess` to false on the session and on every request. Wi-Fi only
  or charging only make the session discretionary, so iOS may wait for Wi-Fi and power. A changed
  option takes effect at the next request.
- Stop on the Sync screen cancels the transfer in flight. The next sync continues it from the
  server's offset.

## Building (needs a Mac)

The app cannot be built or run on the server; it needs Xcode 15 or later and an Apple developer
account for a real device.

```bash
brew install xcodegen
cd ios
xcodegen generate
open PersonalContentCloud.xcodeproj
```

In Xcode: pick your team under Signing and Capabilities (the bundle id `in.rivant.pcc` can be changed in
`project.yml`), make sure Background Modes lists "Background fetch" and "Background processing", then run
on your iPhone. A simulator has no real photo library to speak of, so test on a device.

## First run

1. On the website, open Settings > Sessions > Devices, add a device named after your phone and copy the
   token (it is shown once).
2. In the app, enter the server address (the https address you use in the browser) and the token.
3. Allow full access to Photos when asked. The first sync checks the whole library against the server
   without uploading what it already has.

## Testing the background uploads on a device

1. Build and run from Xcode on an iPhone, sign in, and turn off "Only on Wi-Fi" if you test on
   cellular.
2. Start a sync with a large video in the library. While it is sending, press the Home button. The
   upload keeps going: `GET /api/uploads/mine?client_id=ios`, opened in a browser signed in to the
   server, shows `received_bytes` growing.
3. Kill the app from the app switcher while a video is sending, then open it again. The Sync screen
   shows "Continuing ..." and the server's upload continues from its last offset. It is not
   created again.
4. Turn on airplane mode in the middle of an upload, wait a minute, turn it off. The transfer
   resumes after the backoff, with no new upload on the server.
5. On the website, move a synced photo to the trash. In the app, open Delete synced and pull to
   refresh. The photo is listed under "In your cloud's trash", and "All" does not select it.
6. To see the delegate relaunch in Xcode: Debug > Attach to Process by PID or Name, enter the app's
   name, start an upload, then stop the app with the Stop button in Xcode, not from the app
   switcher. A force quit from the switcher cancels background transfers, and the next sync then
   continues them from the server's offset.

Xcode's console logs the background session's events. The app's own log is on the Sync screen.

## Limits

- iOS decides when background work runs; a large first backup finishes fastest with the app open and
  the phone plugged in.
- A file keeps uploading while the app is suspended. The next asset is exported and started on the
  next run: when the app opens, when the library changes, or in the background processing window.
  Exporting needs the app to run, because originals in iCloud are downloaded first.
- A force quit from the app switcher cancels the background transfers (an iOS rule). The next run
  continues them from the server's offset.
- Deleting from the library always goes through the system confirmation; iOS does not allow silent
  deletion.
