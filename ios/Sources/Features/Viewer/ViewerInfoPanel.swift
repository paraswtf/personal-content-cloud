import SwiftUI
import MapKit

/// The viewer's info panel (the website's components/viewer/InfoPanel.tsx): the vault note, a possible mis-click,
/// how it was received, the processing notes, the boxes area's people section, maybe related to, tags, the virtual
/// document, category, labels, quality, the text read from it, albums, the file's details with its place, and the
/// item's actions. The core shows it as a sheet; it reads the item again after each change, since the sheet keeps
/// the context it was opened with.
struct ViewerInfoPanel: View {
    let context: ViewerPageContext
    let boxes: ViewerBoxesModel

    @Environment(AppRouter.self) private var router
    @State private var fresh: ViewerPageContext?
    @State private var confirmTrash = false
    @State private var trashDays = 30
    @State private var file: ViewerFile?
    @State private var downloading = false

    private var ctx: ViewerPageContext { fresh ?? context }

    var body: some View {
        Group {
            if let item = ctx.item {
                panel(item)
            } else {
                DefaultInfoPanel(context: ctx)
            }
        }
        .onCloudChange(uid: context.uid) { Task { await refresh() } }
        .task { trashDays = await ViewerConfig.loadTrashDays() }
        .sheet(item: $file) { file in
            ViewerShareSheet(items: [file.url]).onDisappear { ViewerFiles.discard(file) }
        }
        // the viewer's own toasts are under this sheet: show them here too
        .toastHost()
    }

    private func refresh() async {
        if let again = await ViewerActions.reload(ctx) { fresh = again }
    }

    // MARK: the panel

    private func panel(_ item: API.ItemDetail) -> some View {
        let kind = MediaKind(item.kind.rawValue)
        let taken = item.takenAt != nil ? "Taken \(Fmt.date(item.takenAt, withTime: true))"
                                        : "Added \(Fmt.date(item.createdAt))"
        return List {
            Group {
                Section {
                    Text(taken).foregroundStyle(.secondary)
                    if item.nsfw {
                        Label("Private. Only shown in the vault.", systemImage: "eye.slash")
                    }
                }
                if item.misclick { misclickSection(item) }
                if ViewerFormat.receivedLabel(item.receivedVia) != nil || item.forward { forwardSection(item) }
                processingSection(item, kind: kind)
                ViewerBoxesInfoSection(model: boxes, context: ctx)
                ViewerRelatedSection(context: ctx)
                ViewerTagsSection(context: ctx, item: item)
                if let document = item.document {
                    Section("Document") {
                        ViewerDocumentCard(context: ctx, document: document)
                    }
                }
            }
            Group {
            if !item.nsfw { categorySection(item) }
            if !item.labels.isEmpty {
                Section("Labels") {
                    ViewerFlow {
                        ForEach(item.labels, id: \.label) { label in
                            Chip(text: label.label)
                                .accessibilityLabel("\(label.label), \(Fmt.percent(label.score))")
                        }
                    }
                }
            }
            qualitySection(item)
            if item.textStatus == .done {
                Section("Text") { ViewerTextSection(context: ctx) }
            }
            if item.deletedAt == nil {
                ViewerAlbumsSection(context: ctx, item: item)
            }
            detailsSection(item, kind: kind)
            actionsSection(item, kind: kind)
            }
        }
        .accessibilityIdentifier("viewer.info.list")
        .navigationTitle(ViewerFormat.detailsTitle(kind))
        .navigationBarTitleDisplayMode(.inline)
        .confirmAction("Move this \(ViewerFormat.trashNoun(kind)) to the trash?", isPresented: $confirmTrash,
                       actionTitle: "Move to trash",
                       message: "It leaves the library now and is deleted for good after \(trashDays) days. Until then you can restore it from Trash with its tags and document details.") {
            Task { await ViewerActions.trash(ctx, router: router) }
        }
    }

    // MARK: sections

    private func misclickSection(_ item: API.ItemDetail) -> some View {
        Section("Possible mis-click") {
            ViewerFlow {
                ForEach(ViewerFormat.reasonKeys(item.misclickReasons), id: \.self) { key in
                    Chip(text: ViewerFormat.reasonLabel(key), systemImage: "exclamationmark.triangle", tint: .orange)
                }
            }
            if let best = ViewerFormat.burstBestUid(item.misclickReasons), best != item.uid {
                Button("A sharper shot of this burst is here") {
                    router.viewer = ViewerRequest(entries: [ViewerEntry(uid: best)], start: 0,
                                                  vault: ctx.vault || item.nsfw)
                }
            }
            HStack(spacing: 12) {
                Button("Keep") {
                    Task {
                        await ViewerActions.run(ctx, "Kept. It is no longer flagged.", doing: "save the change") {
                            _ = try await $0.updateItem(uid: item.uid, body: API.ItemUpdate(misclick: false))
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier("viewer.info.keep")
                Button("Move to trash", role: .destructive) { confirmTrash = true }
                    .buttonStyle(.bordered)
            }
        }
    }

    private func forwardSection(_ item: API.ItemDetail) -> some View {
        Section(ViewerFormat.receivedLabel(item.receivedVia) ?? "Forward") {
            if item.forward {
                Text("Set apart as a forward. It is not in the library unless forwards are included.")
                let reasons = ViewerFormat.forwardReasons(item.forwardReasons)
                if !reasons.isEmpty {
                    ViewerFlow {
                        ForEach(reasons, id: \.self) { Chip(text: $0, systemImage: "arrowshape.turn.up.right", tint: .blue) }
                    }
                }
                HStack(spacing: 12) {
                    Button("Not a forward") {
                        Task {
                            await ViewerActions.run(ctx, "Kept. It stays in the library.", doing: "save the change") {
                                _ = try await $0.updateItem(uid: item.uid, body: API.ItemUpdate(forward: false))
                            }
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    Button("Move to trash", role: .destructive) { confirmTrash = true }
                        .buttonStyle(.bordered)
                }
            } else {
                Text("Kept in the library as a photo\(item.forwardSource == .user ? ", as you decided" : "").")
                Button {
                    Task {
                        await ViewerActions.run(ctx, "Set apart as a forward.", doing: "save the change") {
                            _ = try await $0.updateItem(uid: item.uid, body: API.ItemUpdate(forward: true))
                        }
                    }
                } label: {
                    Label("Mark as forward", systemImage: "arrowshape.turn.up.right")
                }
                .accessibilityIdentifier("viewer.info.markForward")
            }
        }
    }

    @ViewBuilder
    private func processingSection(_ item: API.ItemDetail, kind: MediaKind) -> some View {
        let notes = processingNotes(item, kind: kind)
        if !notes.isEmpty {
            Section("Processing") {
                ForEach(notes, id: \.text) { note in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        if note.busy {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "exclamationmark.circle").foregroundStyle(.orange)
                        }
                        Text(note.text)
                    }
                }
            }
        }
    }

    private struct Note: Hashable {
        let text: String
        let busy: Bool
    }

    private func processingNotes(_ item: API.ItemDetail, kind: MediaKind) -> [Note] {
        var notes: [Note] = []
        if item.analysisStatus == .pending { notes.append(Note(text: "Looking for faces, people and categories.", busy: true)) }
        if item.analysisStatus == .error {
            notes.append(Note(text: "Analysis failed for this item. Try \"Analyse again\".", busy: false))
        }
        if item.textStatus == .pending { notes.append(Note(text: "Reading the text.", busy: true)) }
        if item.textStatus == .error {
            notes.append(Note(text: "The text could not be read. Try \"Read text again\".", busy: false))
        }
        if kind == .video && item.previewStatus == .pending {
            notes.append(Note(text: "A lighter version of this video is being prepared.", busy: true))
        }
        if kind == .video && item.previewStatus == .error {
            notes.append(Note(text: "The web version of this video could not be made. The original is still available.",
                              busy: false))
        }
        return notes
    }

    private func categorySection(_ item: API.ItemDetail) -> some View {
        Section("Category") {
            Picker("Category", selection: Binding(
                get: { item.category },
                set: { value in
                    guard value != item.category else { return }
                    Task {
                        await ViewerActions.run(ctx, "Category set to \(ViewerFormat.categoryLabel(value.rawValue))",
                                                doing: "save the change") {
                            _ = try await $0.updateItem(uid: item.uid, body: API.ItemUpdate(category: value))
                        }
                    }
                })) {
                ForEach(ViewerFormat.categories, id: \.self) { category in
                    Text(ViewerFormat.categoryLabel(category.rawValue)).tag(category)
                }
            }
            .pickerStyle(.menu)
            .accessibilityIdentifier("viewer.info.category")
            Text(item.categorySource == .user ? "Set by you" : "Chosen automatically")
                .font(.footnote)
                .foregroundStyle(.secondary)
            ForEach(ViewerFormat.topScores(item.categoryScores), id: \.key) { score in
                HStack(spacing: 8) {
                    Text(ViewerFormat.categoryLabel(score.key))
                        .font(.caption)
                        .frame(width: 92, alignment: .leading)
                    ProgressView(value: max(0.02, min(1, score.value)))
                    Text(Fmt.percent(score.value))
                        .font(.caption.monospacedDigit())
                        .frame(width: 44, alignment: .trailing)
                }
                .accessibilityElement(children: .combine)
            }
        }
    }

    @ViewBuilder
    private func qualitySection(_ item: API.ItemDetail) -> some View {
        let lines = ViewerFormat.qualityLines(item.qualityScores)
        if !lines.isEmpty {
            Section("Quality") {
                ForEach(lines, id: \.label) { LabeledContent($0.label, value: $0.value) }
            }
        } else if ViewerFormat.qualitySkipped(item.qualityScores) {
            Section("Quality") {
                Text("Quality checks were skipped for this item.").foregroundStyle(.secondary)
            }
        }
    }

    private func detailsSection(_ item: API.ItemDetail, kind: MediaKind) -> some View {
        Section("Details") {
            row("Date taken", Fmt.date(item.takenAt, withTime: true))
            row("Added", Fmt.date(item.createdAt, withTime: true))
            row("File name", item.filename)
            row("Type", item.mime)
            row("Dimensions", Fmt.resolution(item.width, item.height))
            row("Size", Fmt.bytes(item.sizeBytes))
            row("Pages", item.pageCount.map { $0 > 0 ? String($0) : "" } ?? "")
            row("Camera", item.camera ?? "")
            if let gps = item.gps {
                ViewerLocationRow(lat: gps.lat, lon: gps.lon, place: item.place)
            }
            if kind == .video {
                row("Length", Fmt.duration(item.durationS))
                row("Video", ViewerFormat.videoLine(codec: item.videoCodec, fps: item.fps, bitrate: item.bitrate,
                                                    hasAudio: item.hasAudio))
            }
        }
    }

    @ViewBuilder
    private func row(_ label: String, _ value: String) -> some View {
        if !value.isEmpty { LabeledContent(label, value: value) }
    }

    private func actionsSection(_ item: API.ItemDetail, kind: MediaKind) -> some View {
        Section("Actions") {
            Button {
                Task { await ViewerActions.setFavourite(ctx, !item.favorite) }
            } label: {
                Label(item.favorite ? "Unfavourite" : "Favourite", systemImage: item.favorite ? "star.fill" : "star")
            }
            .accessibilityIdentifier("viewer.info.favourite")
            if let download = item.urls.download {
                Button {
                    Task { await fetchOriginal(item, address: download) }
                } label: {
                    HStack {
                        Label("Download original", systemImage: "arrow.down.circle")
                        if downloading { Spacer(); ProgressView() }
                    }
                }
                .disabled(downloading)
            }
            Button {
                Task { await ViewerActions.setPrivate(ctx, !item.nsfw, router: router) }
            } label: {
                Label(item.nsfw ? "Not private" : "Mark private", systemImage: item.nsfw ? "lock.open" : "lock")
            }
            Button {
                Task { await ViewerActions.reprocess(ctx, .analyze) }
            } label: {
                Label("Analyse again", systemImage: "arrow.clockwise")
            }
            .disabled(item.analysisStatus == .pending)
            if kind == .pdf || kind == .document || item.textStatus != .none {
                Button {
                    Task { await ViewerActions.reprocess(ctx, .text) }
                } label: {
                    Label("Read text again", systemImage: "arrow.clockwise")
                }
                .disabled(item.textStatus == .pending)
            }
            if kind == .video {
                Button {
                    Task { await ViewerActions.reprocess(ctx, .preview) }
                } label: {
                    Label("Remake video copy", systemImage: "arrow.clockwise")
                }
                .disabled(item.previewStatus == .pending)
            }
            Button(role: .destructive) {
                confirmTrash = true
            } label: {
                Label("Move to trash", systemImage: "trash")
            }
            .accessibilityIdentifier("viewer.info.trash")
        }
    }

    private func fetchOriginal(_ item: API.ItemDetail, address: String) async {
        downloading = true
        defer { downloading = false }
        do {
            file = try await ViewerFiles.fetch(ctx, address: address, filename: item.filename)
        } catch {
            Toaster.shared.error(error, doing: "download the original")
        }
    }
}

/// The place of an item: its name (or its coordinates), a small map, Show on map (the website's LocationRow: the
/// app's own map at this spot, listing what is there) and Open in Maps (Apple Maps, for directions).
struct ViewerLocationRow: View {
    let lat: Double
    let lon: Double
    let place: String?

    @Environment(AppRouter.self) private var router

    private var coordinate: CLLocationCoordinate2D { CLLocationCoordinate2D(latitude: lat, longitude: lon) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            LabeledContent("Location", value: place ?? Fmt.coords(lat: lat, lon: lon))
            Map(initialPosition: .region(MKCoordinateRegion(center: coordinate,
                                                            span: MKCoordinateSpan(latitudeDelta: 0.05,
                                                                                   longitudeDelta: 0.05)))) {
                Marker(place ?? "", coordinate: coordinate)
            }
            .frame(height: 150)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .allowsHitTesting(false)
            .accessibilityHidden(true)
            HStack(spacing: 20) {
                Button {
                    // The viewer closes and the map opens on the tab underneath, as the website's link leaves the item.
                    router.viewer = nil
                    router.open(.mapAt(MapStart(latitude: lat, longitude: lon)))
                } label: {
                    Label("Show on map", systemImage: "map")
                }
                .buttonStyle(.borderless)
                .accessibilityIdentifier("viewer.info.showOnMap")
                Button {
                    let item = MKMapItem(placemark: MKPlacemark(coordinate: coordinate))
                    item.name = place
                    item.openInMaps(launchOptions: [MKLaunchOptionsMapCenterKey: NSValue(mkCoordinate: coordinate)])
                } label: {
                    Label("Open in Maps", systemImage: "arrow.up.forward.app")
                }
                .buttonStyle(.borderless)
                .accessibilityIdentifier("viewer.info.openInMaps")
            }
        }
    }
}

/// Chips that wrap to the next line when the row is full.
struct ViewerFlow<Content: View>: View {
    var spacing: CGFloat = 8
    @ViewBuilder let content: () -> Content

    var body: some View {
        ViewerFlowLayout(spacing: spacing).callAsFunction { content() }
    }
}

/// The layout behind ViewerFlow.
struct ViewerFlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 320
        var x: CGFloat = 0, y: CGFloat = 0, row: CGFloat = 0, widest: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(ProposedViewSize(width: width, height: nil))
            if x > 0 && x + size.width > width {
                x = 0
                y += row + spacing
                row = 0
            }
            x += size.width + spacing
            widest = max(widest, x - spacing)
            row = max(row, size.height)
        }
        return CGSize(width: proposal.width ?? widest, height: y + row)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, row: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(ProposedViewSize(width: bounds.width, height: nil))
            if x > bounds.minX && x + size.width > bounds.maxX {
                x = bounds.minX
                y += row + spacing
                row = 0
            }
            view.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            row = max(row, size.height)
        }
    }
}

/// The server's settings the viewer needs (how long the trash keeps items), asked for once.
@MainActor
enum ViewerConfig {
    static var trashDays = 30
    private static var loaded = false

    static func loadTrashDays() async -> Int {
        if loaded { return trashDays }
        if let config = try? await Cloud.api().getConfig() {
            trashDays = config.trashDays
            loaded = true
        }
        return trashDays
    }
}

extension ViewerActions {
    static func setFavourite(_ context: ViewerPageContext, _ on: Bool) async {
        await run(context, on ? "Added to favourites" : "Removed from favourites", doing: "save the change") {
            _ = try await $0.updateItem(uid: context.uid, body: API.ItemUpdate(favorite: on))
        }
    }

    /// Moves the item into or out of the vault; it then leaves the list the viewer was opened on, so the viewer closes.
    static func setPrivate(_ context: ViewerPageContext, _ on: Bool, router: AppRouter) async {
        let done = await run(context, on ? "Moved to the vault" : "Moved out of the vault", doing: "save the change") {
            _ = try await $0.updateItem(uid: context.uid, body: API.ItemUpdate(nsfw: on))
        }
        if done { router.viewer = nil }
    }

    static func reprocess(_ context: ViewerPageContext, _ stage: API.Stages) async {
        await run(context, "Processing again. This can take a minute.", doing: "start processing", alwaysSay: true) {
            _ = try await $0.reprocess(uid: context.uid, body: API.ReprocessIn(stages: [stage]))
        }
    }

    /// Moves the item to the trash (the message offers Undo) and closes the viewer.
    static func trash(_ context: ViewerPageContext, router: AppRouter) async {
        let done = await run(context, "Moved to the trash", doing: "move it to the trash") {
            _ = try await $0.deleteItem(uid: context.uid)
        }
        if done { router.viewer = nil }
    }
}
