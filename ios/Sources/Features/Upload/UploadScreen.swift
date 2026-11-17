import SwiftUI
import SwiftData
import UIKit
import PhotosUI
import UniformTypeIdentifiers

/// Upload and sync (the website's pages/UploadPage.tsx, with the phone's own sync below it). Photos and videos from
/// the Photos picker, any file from Files or dropped on the screen, and the camera, into the upload queue
/// (UploadQueue), whose files keep uploading in the background; then the phone's library sync, removing synced
/// photos from the phone, and the sync's options. Opening it is where the app first asks for Photos access (never
/// at launch or sign-in), and where the queue continues what it left when the app was closed.
struct UploadScreen: View {
    @Environment(AppRouter.self) private var router
    @ObservedObject private var uploader = BackgroundUploader.shared
    @ObservedObject private var sync = SyncEngine.shared
    @Query(UploadScreen.queueDescriptor) private var rows: [SyncRecord]

    @State private var picked: [PhotosPickerItem] = []
    @State private var choosingFiles = false
    @State private var cameraOpen = false
    @State private var cameraRefused = false
    @State private var loadingConfig = false
    @State private var configError: Error?

    private let queue = UploadQueue.shared

    static var queueDescriptor: FetchDescriptor<SyncRecord> {
        FetchDescriptor<SyncRecord>(predicate: #Predicate<SyncRecord> { $0.queuedAt != nil },
                                    sortBy: [SortDescriptor(\.queuedAt)])
    }

    var body: some View {
        let infos = rows.map { queue.info($0, progress: uploader.sending[$0.assetId]?.progress) }
        List {
            addSection
            if !rows.isEmpty {
                queueSection(infos)
            }
            syncSection
            notesSection
        }
        .navigationTitle("Upload and sync")
        .dropDestination(for: UploadPickedFile.self) { files, _ in
            queue.add(files.map { UploadIncoming(url: $0.url, name: $0.name) }, origin: "file")
            return !files.isEmpty
        }
        .onChange(of: picked) { _, items in
            guard !items.isEmpty else { return }
            picked = []
            Task { await queue.add(picked: items) }
        }
        .fileImporter(isPresented: $choosingFiles, allowedContentTypes: [.item], allowsMultipleSelection: true) { result in
            addChosenFiles(result)
        }
        .fullScreenCover(isPresented: $cameraOpen) {
            UploadCameraPicker(onCapture: { queue.add([$0], origin: "camera") },
                               onError: { Toaster.shared.error($0) },
                               onClose: { cameraOpen = false })
                .ignoresSafeArea()
        }
        .alert("Camera access was refused", isPresented: $cameraRefused) {
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Allow the camera for Content Cloud in the Settings app.")
        }
        // Opening upload is where the app first asks for Photos access (never at launch or sign-in).
        .task {
            await PhotoLibrary.requestAccessIfNeeded()
            await loadConfig()
            await queue.resume()
        }
        // back online: files left waiting go on, and files that failed for the network start again
        .onChange(of: sync.online) { _, online in
            if online { queue.backOnline() }
        }
        .onDisappear { queue.stopPolling() }
    }

    private func queueSection(_ infos: [UploadRowInfo]) -> some View {
        let summary = UploadSummary.of(infos)
        let listRows = zip(rows, infos).map { UploadListRow(rec: $0.0, info: $0.1) }
        return Section {
            UploadSummaryRow(summary: summary, progress: UploadSummary.progress(infos),
                             onClearFinished: { queue.clearFinished() })
            ForEach(listRows) { row in
                UploadQueueRow(rec: row.rec, info: row.info, resumed: queue.resumed.contains(row.rec.assetId),
                               offline: !sync.online, retryAt: uploader.retryAt[row.rec.assetId],
                               onView: viewAction(row.rec, row.info),
                               onRetry: { queue.retry(row.rec) },
                               onCancel: { queue.cancel(row.rec) },
                               onRemove: { queue.remove(row.rec) })
            }
        } header: {
            Text("Upload queue")
        } footer: {
            if summary.total > 0 && summary.sending == 0 && summary.processing == 0 && summary.added > 0 {
                VStack(alignment: .leading, spacing: 6) {
                    Text("New files are in the library. Sorting and indexing may take a little longer.")
                    Button("Open the library") { router.tab = .library }
                        .font(.footnote.weight(.semibold))
                        .accessibilityIdentifier("upload.openLibrary")
                }
            }
        }
    }

    private var syncSection: some View {
        Section("This phone's photos") {
            NavigationLink { SyncView() } label: { Label("Sync", systemImage: "arrow.triangle.2.circlepath") }
                .accessibilityIdentifier("upload.sync")
            NavigationLink { DeleteSyncedView() } label: { Label("Delete synced", systemImage: "trash") }
                .accessibilityIdentifier("upload.deleteSynced")
            NavigationLink { SettingsView() } label: { Label("Sync settings", systemImage: "slider.horizontal.3") }
                .accessibilityIdentifier("upload.syncSettings")
        }
    }

    private var notesSection: some View {
        Section {
            ForEach(UploadText.notes, id: \.text) { note in
                Label {
                    Text(note.text).font(.subheadline).foregroundStyle(.secondary)
                } icon: {
                    Image(systemName: note.systemImage).foregroundStyle(.tint)
                }
            }
        } footer: {
            // the limit only once the server said it (here, below everything, so reading it moves no row the owner
            // is about to tap)
            Text((queue.configLoaded ? "Up to \(UploadText.limit(queue.maxUploadBytes)) per file. "
                  : loadingConfig ? "Reading the server's upload limit. " : "")
                 + "Photos (including HEIC and RAW), videos, PDFs, Word, Excel, PowerPoint and text files are indexed. Originals are kept exactly as uploaded.")
        }
    }

    private var addSection: some View {
        Section {
            Text("Add photos, videos and documents from this device, or take them with the camera.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            PhotosPicker(selection: $picked, matching: .any(of: [.images, .videos]), preferredItemEncoding: .current,
                         photoLibrary: .shared()) {
                Label("Choose photos and videos", systemImage: "photo.on.rectangle.angled")
            }
            .accessibilityIdentifier("upload.photos")
            Button {
                choosingFiles = true
            } label: {
                Label("Choose files", systemImage: "folder")
            }
            .accessibilityIdentifier("upload.files")
            Button {
                if UploadCamera.refused { cameraRefused = true } else { cameraOpen = true }
            } label: {
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Take a photo or video")
                        if !UploadCamera.available {
                            Text("This device has no camera.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                } icon: {
                    Image(systemName: "camera")
                }
            }
            .disabled(!UploadCamera.available)
            .accessibilityIdentifier("upload.camera")
            if UploadCamera.available && UploadCamera.microphoneRefused {
                Label("No microphone: clips are recorded without sound.", systemImage: "mic.slash")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("upload.noMicrophone")
            }
            configRows
            #if DEBUG
            if UploadTestHook.files != nil {
                Button("Add the test files") { UploadTestHook.add() }
                    .accessibilityIdentifier("upload.testFiles")
            }
            #endif
            if queue.importing > 0 {
                HStack(alignment: .top, spacing: 8) {
                    ProgressView()
                    Text("Waiting for the files. Photos kept in iCloud are downloaded before they are handed over, which can take a while.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            if let notice = queue.notice {
                Text(notice)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("upload.notice")
            }
        }
    }

    /// What went wrong reading the server's upload settings (its size limit per file), with Try again (the website's
    /// error card on its Upload page); while they are read, the notes' footer says so. The rest of the screen stays
    /// usable meanwhile.
    @ViewBuilder private var configRows: some View {
        if let configError {
            VStack(alignment: .leading, spacing: 6) {
                Label(ErrorText.describe(configError, doing: "read the server's upload settings"),
                      systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundStyle(.orange)
                    .accessibilityIdentifier("upload.configError")
                Button("Try again") { Task { await loadConfig() } }
                    .font(.footnote.weight(.semibold))
                    .disabled(loadingConfig)
                    .accessibilityIdentifier("upload.configRetry")
            }
        }
    }

    private func loadConfig() async {
        loadingConfig = true
        defer { loadingConfig = false }
        do {
            let config = try await Cloud.api().getConfig()
            queue.maxUploadBytes = Int64(config.maxUploadBytes)
            queue.configLoaded = true
            configError = nil
        } catch {
            if error is CancellationError || (error as? URLError)?.code == .cancelled { return }
            configError = error
        }
    }

    /// View, for a stored file or one already in the library: the item in the viewer (with vault=1 for a vault item).
    private func viewAction(_ rec: SyncRecord, _ info: UploadRowInfo) -> (() -> Void)? {
        guard info.phase == .done, rec.serverStatus == "stored" || rec.serverStatus == "duplicate", let uid = rec.uid else {
            return nil
        }
        let thumb = rec.thumbRef
        let vault = rec.itemVault
        return {
            router.viewer = ViewerRequest(entries: [ViewerEntry(uid: uid, thumb: thumb)], start: 0, vault: vault)
        }
    }

    /// Files chosen in Files: copied into the app first (their addresses only last while the picker's answer is
    /// handled), then queued.
    private func addChosenFiles(_ result: Result<[URL], Error>) {
        guard case .success(let urls) = result, !urls.isEmpty else { return }
        queue.importing += urls.count
        Task {
            var copies: [UploadIncoming] = []
            var unreadable = 0
            for url in urls {
                if let copy = try? await UploadIncoming.copy(fromPicked: url) { copies.append(copy) } else { unreadable += 1 }
                queue.importing = max(0, queue.importing - 1)
            }
            queue.add(copies, origin: "file", unreadable: unreadable)
        }
    }
}

/// A queued record with what its row shows.
struct UploadListRow: Identifiable {
    let rec: SyncRecord
    let info: UploadRowInfo

    var id: String { rec.assetId }
}
