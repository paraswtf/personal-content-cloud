# The iPhone app: structure and conventions

The app is SwiftUI for iOS 17 (Observation, no third-party packages), generated into an Xcode project by
XcodeGen from `ios/project.yml` and built on GitHub's macOS machines (`.github/workflows/ios.yml`). The server's
contract is `docs/ARCHITECTURE.md`; the website (`frontend/src`) is the reference for what each screen does.

## Folders

| Folder | What it holds | Who changes it |
|---|---|---|
| `Sources/API/` | The generated API layer (`Models.generated.swift`, `Endpoints.generated.swift`) and its transport (`APIClient+Core.swift`). Regenerate with `backend/.venv/bin/python scripts/gen_swift_api.py`; `backend/tests/test_swift_api.py` fails when the files are stale. | The generator only |
| `Sources/Core/` | Services: the connection, the core's own requests, images, video, paging, undo, toasts and error text, formatting, routes, the adapters from API types to the core's shapes. | Core work |
| `Sources/UI/` | Shared views: the media grid, the viewer core and its boxes, the person picker, selection and the bulk bar, avatar, chips, states. | Core work |
| `Sources/Shell/` | The tabs, each tab's navigation stack, the More menu, the routes' destinations. | Core work |
| `Sources/Features/<Area>/` | One folder per area; everything an area needs beyond the core lives here. | That area only |
| `Sources/Views/`, `Sources/Services/`, `Sources/Models/` | The sign-in screen and the phone library sync (SyncEngine, BackgroundUploader, SyncView, DeleteSyncedView, sync settings). | Upload area and core |

## The areas and their entry files

Each area replaces its placeholder files (keep the type names and initializers, the shell calls them):

| Area | Files the shell calls | Website reference |
|---|---|---|
| Library | `Features/Library/LibraryScreen.swift`: `LibraryScreen()` | `pages/LibraryPage.tsx`, `components/library`, `components/timeline` |
| Viewer | `Features/Viewer/ViewerScreen.swift`: `ViewerScreen(request:)` | `components/Viewer.tsx`, `components/viewer` |
| People | `Features/People/PeopleScreen.swift`: `PeopleScreen()`; `PersonScreen.swift`: `PersonScreen(personId:)` | `pages/PeoplePage.tsx`, `pages/PersonPage.tsx`, `components/people` |
| Review | `Features/Review/ReviewScreen.swift`: `ReviewScreen()` | `pages/ReviewPage.tsx` |
| Albums | `Features/Albums/AlbumsScreen.swift`: `AlbumsScreen()`; `AlbumScreen.swift`: `AlbumScreen(albumId:)` | `pages/AlbumsPage.tsx`, `pages/AlbumPage.tsx`, `components/albums`, `components/share` |
| Map | `Features/Map/MapScreen.swift`: `MapScreen()` | `pages/MapPage.tsx`, `components/map` |
| Documents | `Features/Documents/DocumentsScreen.swift`: `DocumentsScreen()` | `pages/DocumentsPage.tsx`, `components/documents` |
| Vault | `Features/Vault/VaultScreen.swift`: `VaultScreen()` | `pages/VaultPage.tsx`, `components/vault` |
| Upload | `Features/Upload/UploadScreen.swift`: `UploadScreen()` (keep its links to SyncView, DeleteSyncedView and SettingsView) | `pages/UploadPage.tsx`, `components/upload` |
| Trash | `Features/Trash/TrashScreen.swift`: `TrashScreen()` | `pages/TrashPage.tsx`, `components/trash` |
| Settings | `Features/Settings/SettingsScreen.swift`: `SettingsScreen()` | `pages/SettingsPage.tsx`, `components/shell/settings`, `components/admin` |

Type names must be unique across the app: prefix an area's own types with the area (`PeopleFilterBar`,
`AlbumRow`), and do not add types to `Core`, `UI` or `Shell` from an area. Something two areas need goes to
the core in its own change.

## Calling the server

- Screens use the generated endpoints: `let page = try await Cloud.api().listItems(cursor: cursor, limit: 120)`.
  `Cloud.api()` (main actor) gives the client for the signed-in server and throws a 401 when signed out. Types
  are in the `API` namespace (`API.ItemCard`, `API.PersonOut`); properties are lowerCamelCase; string enums have
  an `unknown(String)` case, so switch with a `default`.
- The vault: wrap the calls of a vault screen in `APIContext.$vault.withValue(true) { ... }`, and pass
  `vault: true` to `RemoteImage` and `AuthorizedPlayer` for vault items' pictures and videos.
- The core's own small requests (`CoreHTTP`, with snake_case decoding into the `Core...` structs) are for the core;
  areas do not need them.

## Lists

- `Pager<Item>` loads a list a page at a time: `Pager { cursor in try await Cloud.api().listTrash(cursor: cursor).pagerPage }`
  (`API.ItemPage.pagerPage` gives a page; build `Pager.Page(items:next:total:)` for other shapes). Call
  `await pager.refresh()` in `.task` when `!pager.loaded`, `pager.loadMoreIfNeeded(item)` as rows appear, and use
  `PagerStateView` for the loading, failed and empty states.
- `MediaGrid(items:selection:onOpen:onAppear:)` shows any `MediaItem` (`API.ItemCard`, `API.SearchHit`, or an
  adapter of your own). `SelectionModel<String>` and `BulkBar` handle selecting.
- Refresh when the server changed: `.onCloudChange { Task { await pager.refresh() } }` (posted after an undo or a
  redo, and by any screen with `CloudEvents.post(items:)` after a change other screens show).

## Writes, undo and messages

- Every owner action goes through `UndoCenter.shared.perform("Named the face") { try await api.assignFaces(...) }`.
  All writes inside share one action id, so a loop over many items undoes as one; when the server changed
  something, the action joins the history and the toast offers Undo (with the website's wording for refusals).
  The label is the success message in plain words, without the names of people in the library in code.
  Writes made outside `perform` still join the history with a plain label, but show no message.
- Errors: `Toaster.shared.error(error, doing: "save the name")` shows "Could not save the name. ..." with the
  server's code explained (`ErrorText`). A cancelled request shows nothing.
- `Toaster.shared.ok("Added to the album")` for a plain confirmation.

## Navigation

- `NavigationLink(value: Route.person(id: 3))` pushes on the current tab; `router.open(.album(id: 2))` does the same
  from code (`@Environment(AppRouter.self) private var router`).
- Items open in the viewer over everything: `router.show(items, start: item, loadMore: { ... })` with the list the
  item came from, so the owner can page through it. `router.open(.item(uid:))` opens a single item.
- The viewer core (`ViewerCore(request:hooks:)`) pages, zooms and plays; the viewer area fills `ViewerHooks`:
  `onBoxTap` (the box under the finger, always the smallest one there), `infoPanel`, `toolbar`, and
  `selectedBoxId` (drawn selected, never raised above a smaller box). `ViewerPageContext.item` is the whole
  `API.ItemDetail`; call `context.reload()` after a change to the item.

## People

- Only named people are shown anywhere in the app: the server already leaves unnamed groups out (a box of one
  arrives with `person` nil, a guess for one arrives as nil). Use `PersonOption(person)`, which is nil for an
  unnamed person, and `Fmt.isNamed(_:)` where a list could still hold one.
- `PersonPickerSheet(title:kind:like:excluding:allowCreate:species:speciesOnly:onPick:)` picks a person (or pet)
  or a new name, as the website's PersonPicker. Pass what is being named as `like:` (`.box(box)`,
  `.faces(ids)`, `.bodies(ids)`, `.animals(ids)`, `.person(id, on: uids)`, `.item(uid)`): with nothing typed, the
  most alike named people come first under "Most alike", each with why ("Looks like them") and "Likely" when the
  matching would take them, then everyone else; typing filters by name and keeps the order of likeness. The list
  waits for the likeness answer, so it never moves under the finger.
- Naming a face of a group the app made in the background makes guesses of the group's other faces: end the
  message with `Fmt.guessedTail(answer.guessed)` ("2 faces that look alike wait in Review."), and a merge's with
  `Fmt.followedTail(answer.followed)`.

## Pictures and video

- `RemoteImage(ref: item.thumbRef, maxPixel: 384)` for any address the API returns (thumbs, previews, face crops,
  covers). The token is sent to the server's own host only; another host's address is not loaded.
- `AuthorizedPlayer.player(for: urls.video)` for videos and Live Photo clips; `VideoPlayer(player:)` or
  `PlayerSurface(player:)` (no controls) to show it.

## Accessibility and words

- Every control has a label (`accessibilityLabel`) and the controls tests use have an `accessibilityIdentifier`
  of the form `area.thing` (`library.select`, `people.search`).
- Words follow the website: plain sentences, no exclamation marks, no em dashes.

## Tests

Screen tests run on the simulator against a seeded server (see the test harness section in `README.md` once it
lands): each area adds one test per screen that opens it and attaches a screenshot.
