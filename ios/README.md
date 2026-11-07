# Content Cloud for iPhone

A small SwiftUI app that keeps your phone's new photos and videos flowing into your Personal Content
Cloud, and that can clear the ones it has synced out of the phone's library when you ask.

What it does:

- Signs in with a device token (Settings > Sessions > Devices on the website creates one; it is shown
  once). No password is stored on the phone.
- Watches the photo library and uploads what is new: photos, videos, and both halves of a Live Photo,
  as the original files, through the server's resumable upload protocol. Nothing is re-sent after a
  reinstall: the app asks the server what it already holds, by asset and by file hash.
- Syncs while the app is open, when the library changes, and in the background when iOS gives it a
  processing window (usually when the phone is charging and idle). Options: Wi-Fi only, charging only,
  include videos.
- "Delete synced": lists the assets the server confirms it holds (it checks again right before) and
  removes them from the library through the system's own confirmation dialog. Nothing is deleted on
  its own.

## Building (needs a Mac)

The app cannot be built or run on the server; it needs Xcode 15 or later and an Apple developer
account for a real device.

```bash
brew install xcodegen
cd ios
xcodegen generate
open PersonalContentCloud.xcodeproj
```

In Xcode: pick your team under Signing and Capabilities (the bundle id `in.biios.pcc` can be changed in
`project.yml`), make sure Background Modes lists "Background fetch" and "Background processing", then run
on your iPhone. A simulator has no real photo library to speak of, so test on a device.

## First run

1. On the website, open Settings > Sessions > Devices, add a device named after your phone and copy the
   token (it is shown once).
2. In the app, enter the server address (the https address you use in the browser) and the token.
3. Allow full access to Photos when asked. The first sync checks the whole library against the server
   without uploading what it already has.

## Limits

- iOS decides when background work runs; a large first backup finishes fastest with the app open and
  the phone plugged in.
- Uploads use a normal URLSession driven by the app and the background task. Switching to a background
  URLSession (uploads that continue while the app is suspended) is the next step.
- Deleting from the library always goes through the system confirmation; iOS does not allow silent
  deletion.
