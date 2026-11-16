import SwiftUI

/// The trash (the website's TrashPage): items deleted in the last `trash_days` days, newest deletion first, as a
/// grid of tiles. Each can be restored (with Undo) or deleted for good after a question; Empty trash deletes
/// everything here for good after a question. Tapping a tile's picture shows it larger (TrashPreview), with the
/// same two buttons and the previous and next items. Private items show a locked tile: their pictures are only
/// served inside the vault. The list leaves vault items out, as on the website.
struct TrashScreen: View {
    @State private var pager = Pager<API.ItemCard> { cursor in
        try await Cloud.api().listTrash(cursor: cursor, limit: 60).pagerPage
    }
    @State private var days = 30
    /// Being restored or deleted: their buttons do nothing until the answer comes.
    @State private var working: Set<String> = []
    /// Items answered since the list was last fetched (the count shown is the server's total less these).
    @State private var answered = 0
    @State private var purging: API.ItemCard?
    @State private var emptying = false
    @State private var emptyBusy = false
    @State private var previewIndex: Int?

    private var total: Int { max((pager.total ?? pager.items.count) - answered, pager.items.count) }

    var body: some View {
        PagerStateView(pager: pager, emptyTitle: TrashText.emptyTitle, emptyImage: "trash",
                       emptyText: TrashText.emptyText(days: days)) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text(TrashText.intro(days: days))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("trash.intro")
                    Text(TrashText.count(total))
                        .font(.subheadline.weight(.medium))
                        .accessibilityIdentifier("trash.count")
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
                        ForEach(Array(pager.items.enumerated()), id: \.element.uid) { index, item in
                            TrashTile(item: item, busy: working.contains(item.uid),
                                      onOpen: { previewIndex = index },
                                      onRestore: { restore(item) },
                                      onPurge: { purging = item })
                                .onAppear { pager.loadMoreIfNeeded(item) }
                        }
                    }
                    if pager.loading && !pager.items.isEmpty {
                        ProgressView().frame(maxWidth: .infinity).padding()
                    }
                }
                .padding(16)
            }
            .refreshable { await refresh() }
        }
        .navigationTitle("Trash")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if !pager.items.isEmpty {
                    Button(role: .destructive) {
                        emptying = true
                    } label: {
                        Label("Empty trash", systemImage: "trash.slash")
                    }
                    .tint(.red)
                    .disabled(emptyBusy)
                    .accessibilityLabel("Empty trash")
                    .accessibilityIdentifier("trash.empty")
                }
            }
        }
        .confirmationDialog("Delete forever", isPresented: purgeAsked, titleVisibility: .visible,
                            presenting: purging) { item in
            Button("Delete forever", role: .destructive) { purge(item) }
            Button("Cancel", role: .cancel) {}
        } message: { item in
            Text(TrashText.purgeQuestion(item.filename))
        }
        .confirmAction("Empty the trash", isPresented: $emptying, actionTitle: "Empty trash",
                       message: TrashText.emptyQuestion(total)) {
            emptyTrash()
        }
        .sheet(isPresented: previewShown) {
            TrashPreview(items: pager.items, index: $previewIndex, total: total,
                         busy: { working.contains($0) },
                         onRestore: { restore($0) },
                         onPurge: { purge($0) },
                         onNearEnd: { item in pager.loadMoreIfNeeded(item) })
        }
        .task {
            if !pager.loaded { await refresh() }
            if let config = try? await Cloud.api().getConfig() { days = config.trashDays }
        }
        .onCloudChange { Task { await refresh() } }
    }

    private var purgeAsked: Binding<Bool> {
        Binding(get: { purging != nil }, set: { if !$0 { purging = nil } })
    }

    private var previewShown: Binding<Bool> {
        Binding(get: { previewIndex != nil }, set: { if !$0 { previewIndex = nil } })
    }

    private func refresh() async {
        await pager.refresh()
        answered = 0
    }

    /// An answered item leaves the list at once (the website keeps it until the list comes again).
    private func drop(_ uid: String) {
        pager.remove { $0.uid == uid }
        answered += 1
    }

    private func restore(_ item: API.ItemCard) {
        guard !working.contains(item.uid) else { return }
        working.insert(item.uid)
        Task {
            defer { working.remove(item.uid) }
            let message = TrashText.restored(item.filename)
            let before = UndoCenter.shared.entries.first?.id
            do {
                try await UndoCenter.shared.perform(message) {
                    try await Cloud.api().restoreItem(uid: item.uid)
                }
                // perform shows the message with Undo when the server recorded the change; say it anyway
                if UndoCenter.shared.entries.first?.id == before { Toaster.shared.ok(message) }
                drop(item.uid)
                CloudEvents.post(items: [item.uid])
            } catch {
                Toaster.shared.error(error, doing: "restore \(item.filename)")
            }
        }
    }

    private func purge(_ item: API.ItemCard) {
        guard !working.contains(item.uid) else { return }
        working.insert(item.uid)
        Task {
            defer { working.remove(item.uid) }
            do {
                // no Undo: the files are gone
                try await UndoCenter.shared.perform(TrashText.purged, toast: false) {
                    try await Cloud.api().purgeOne(uid: item.uid)
                }
                Toaster.shared.ok(TrashText.purged)
                drop(item.uid)
                CloudEvents.post(items: [item.uid])
            } catch {
                Toaster.shared.error(error, doing: "delete \(item.filename)")
            }
        }
    }

    private func emptyTrash() {
        guard !emptyBusy else { return }
        emptyBusy = true
        Task {
            defer { emptyBusy = false }
            do {
                let answer = try await UndoCenter.shared.perform("Emptied the trash", toast: false) {
                    try await Cloud.api().emptyTrash()
                }
                Toaster.shared.ok(TrashText.emptied(answer.updated))
                previewIndex = nil
                await refresh()
                CloudEvents.post()
            } catch {
                Toaster.shared.error(error, doing: "empty the trash")
            }
        }
    }
}

/// One item in the trash: its picture (tap for a larger look), its name, Restore and Delete forever.
struct TrashTile: View {
    let item: API.ItemCard
    let busy: Bool
    let onOpen: () -> Void
    let onRestore: () -> Void
    let onPurge: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: onOpen) {
                Color.clear
                    .aspectRatio(1, contentMode: .fit)
                    .overlay { TrashThumb(item: item, ref: item.urls.thumb, maxPixel: 384) }
                    .clipped()
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Show \(item.filename) larger")
            .accessibilityIdentifier("trash.open")
            Text(item.filename)
                .font(.caption.weight(.medium))
                .lineLimit(1)
                .truncationMode(.middle)
                .padding(.horizontal, 8)
                .padding(.top, 8)
                .accessibilityHidden(true)
            HStack(spacing: 4) {
                Button(action: onRestore) {
                    Label("Restore", systemImage: "arrow.uturn.backward")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity, minHeight: 40)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Restore \(item.filename)")
                .accessibilityIdentifier("trash.restore")
                Button(action: onPurge) {
                    Image(systemName: "trash")
                        .frame(width: 40, height: 40)
                        .contentShape(Rectangle())
                }
                .foregroundStyle(.red)
                .accessibilityLabel("Delete \(item.filename) forever")
                .accessibilityIdentifier("trash.purge")
            }
            .disabled(busy)
            .padding(4)
        }
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(Color(.separator).opacity(0.5)) }
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .opacity(busy ? 0.55 : 1)
        .contextMenu {
            Button(action: onRestore) { Label("Restore", systemImage: "arrow.uturn.backward") }
            Button(role: .destructive, action: onPurge) { Label("Delete forever", systemImage: "trash") }
        }
    }
}

/// An item's picture in the trash: a private item's locked tile, a document's page on white, or the picture,
/// with a video's length.
struct TrashThumb: View {
    let item: API.ItemCard
    let ref: String
    var maxPixel = 384
    var fit = false

    private var kind: MediaKind { MediaKind(item.kind.rawValue) }
    private var isDocument: Bool { kind == .pdf || kind == .document }

    var body: some View {
        ZStack {
            Color(.tertiarySystemFill)
            if item.nsfw {
                VStack(spacing: 4) {
                    Image(systemName: "eye.slash").font(.title2)
                    Text("Private").font(.caption.weight(.medium))
                }
                .foregroundStyle(.secondary)
            } else if isDocument {
                Color.white
                RemoteImage(ref: ref, maxPixel: maxPixel, contentMode: .fit)
                    .padding(4)
            } else {
                RemoteImage(ref: ref, maxPixel: maxPixel, contentMode: fit ? .fit : .fill)
            }
        }
        .overlay(alignment: .bottomTrailing) {
            if kind == .video, let seconds = item.durationS, seconds > 0, !item.nsfw {
                Text(Fmt.duration(seconds))
                    .font(.caption2.weight(.semibold).monospacedDigit())
                    .foregroundStyle(.white)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 6))
                    .padding(6)
            }
        }
        .accessibilityHidden(true)
    }
}
