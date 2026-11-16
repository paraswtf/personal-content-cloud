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

## Building

The server this project is developed on has no Xcode. GitHub's macOS machines build the app instead
(`.github/workflows/ios.yml`): every push that changes `ios/` generates the project with XcodeGen and
builds it for the simulator, so a branch shows whether the app compiles before anyone opens Xcode.

## TestFlight from GitHub

`.github/workflows/ios-testflight.yml` signs the app and uploads it to TestFlight when a push to main
changes `ios/`, and when it is started by hand (Actions > iOS TestFlight > Run workflow; with "upload"
off it signs and exports only, which checks the signing without sending a build). Other branches never
sign. It needs four repository secrets, added under Settings > Secrets and variables > Actions:

| Secret | What it holds |
|---|---|
| `APPLE_TEAM_ID` | the Apple developer team's ID |
| `ASC_KEY_ID` | the Key ID of an App Store Connect API key |
| `ASC_ISSUER_ID` | the Issuer ID shown on the same page |
| `ASC_KEY_P8` | the whole text of that key's `.p8` file |

The key is made in App Store Connect under Users and Access > Integrations > App Store Connect API >
Team Keys, with the Admin role: the export signs with the team's cloud-managed distribution
certificate, which a key with a lower role may not be allowed to create. Apple offers the `.p8` for
download once. Without the four secrets the workflow says so and stops without failing.

The archive itself is not signed; the export signs it and creates the App Store profile, so no
certificate or profile is kept in the repository or in GitHub. Build numbers are the date and time of
the build (`20261117.1530`), so each one is higher than the last whoever uploaded it; an upload made
from a Mac afterwards needs a higher number than the last one from GitHub.

## Building on a Mac

Xcode 15 or later and an Apple developer account for a real device.

```bash
brew install xcodegen
cd ios
xcodegen generate
open PersonalContentCloud.xcodeproj
```

In Xcode: pick your team under Signing and Capabilities (the bundle id `in.rivant.pcc` can be changed in
`project.yml`), make sure Background Modes lists "Background fetch" and "Background processing", then run
on your iPhone. A simulator has no real photo library to speak of, so test on a device.

## Tests on GitHub

The server this project is developed on has no Mac, so the app is compiled and tested by the iOS workflow
(`.github/workflows/ios.yml`) on GitHub's macOS machines, on every push that touches `ios/`. One run:

1. Generates the project with XcodeGen and builds the app and both test targets for the simulator, while an
   iPhone simulator boots in the background (`ci/pick_simulator.py` takes the runtime of the SDK the app is
   built with when the machine has it, else the newest one).
2. Installs the backend into a virtual environment (`ci/install_backend.sh`: the versions of
   `backend/uv.lock`, with the CPU onnxruntime in place of the GPU one) and starts it on 127.0.0.1:8765 with
   the models off (`PCC_NO_MODELS=1`) and a fresh data folder.
3. Seeds it (`ci/seed_server.py`), first with rows and files written by the backend's own test helpers,
   then over HTTP like a client: 36 drawn landscape photos over six months (24 with a place, one in the
   vault, one in the trash, two favourites), two named people ("Test Person", "Test Visitor") with 21 face
   boxes and three faces nobody named, the owner account (set up through `/api/auth/setup`), a device
   token from `POST /api/auth/devices` (the route behind Settings > Sessions > Devices), three photos
   uploaded with the app's resumable protocol (date and place in their EXIF), a text note, two albums
   and a share link of one. Everything is generated; no real people or photos.
4. Runs the unit tests (`UnitTests/`, hosted by the app) and the screen tests (`UITests/`) against that
   server. The seeded address and token reach the tests as `TEST_RUNNER_PCC_TEST_SERVER` and
   `TEST_RUNNER_PCC_TEST_TOKEN` (xcodebuild hands them to the test runner without the prefix); the tests pass
   them to the app as the launch variables `PCC_TEST_SERVER` and `PCC_TEST_TOKEN`, and a Debug build signs
   in with them (`Sources/TestLaunch.swift`, not compiled into Release builds). `PCC_TEST_SIGNED_OUT=1`
   starts on the sign-in screen instead. With a test server the automatic sync is off.
5. Keeps the screenshots as the `screenshots` artifact, the build, test and backend logs as `logs`, and
   on a failure the whole `xcresult` bundle.

A failing test fails the run.

### Adding a screen test

Subclass `ScreenTestCase` in a new file under `UITests/`. `launchSignedIn()` starts the app signed in to
the seeded server, `tab("Name")` taps a tab, and `snap("name")` keeps a screenshot as `name.png` (keep the
names unique across tests):

```swift
final class AlbumsScreens: ScreenTestCase {
    func testAlbums() throws {
        try launchSignedIn()
        tab("Albums")
        XCTAssertTrue(app.staticTexts["Weekend trip"].waitForExistence(timeout: 15))
        snap("albums")
    }
}
```

A screen that needs more in the library gets it in `ci/seed_server.py`: rows and files in `library()`,
anything a client would do in `api()`.

### Reading a run

```bash
gh run list --workflow iOS --branch <branch>
gh run watch <run id> --exit-status
gh run view <run id> --log-failed
gh run download <run id> -n screenshots -D screenshots/
```

`screenshots/tests.txt` lists every test with its result.

### Running the tests on a Mac

Start the backend with the same seed (the commands are in the workflow's "Seed and start the backend"
step), then:

```bash
TEST_RUNNER_PCC_TEST_SERVER=http://127.0.0.1:8765 TEST_RUNNER_PCC_TEST_TOKEN=<token from seed.json> \
  xcodebuild test -project PersonalContentCloud.xcodeproj -scheme PersonalContentCloud \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro'
```

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
