import SwiftUI
import AVKit
import Combine

/// A larger look at one item in the trash (the website's TrashPreview): the preview picture, a video with its
/// controls, or a document's first page, with Restore and Delete forever (after a question), and the previous and
/// next items (the arrows or a swipe). An answered item hands over to the one now in its place; the look closes
/// when the trash is empty. Private items show the same locked tile as the grid.
struct TrashPreview: View {
    let items: [API.ItemCard]
    @Binding var index: Int?
    let total: Int
    let busy: (String) -> Bool
    let onRestore: (API.ItemCard) -> Void
    /// Deletes for good; the second argument asks the question again when deleting failed.
    let onPurge: (API.ItemCard, @escaping () -> Void) -> Void
    /// Shown: the list loads more when it nears the end.
    let onNearEnd: (API.ItemCard) -> Void

    @State private var askPurge: API.ItemCard?

    /// Where the look is: the index, or the last item when the list got shorter.
    private var position: Int? {
        guard let index, !items.isEmpty else { return nil }
        return min(max(0, index), items.count - 1)
    }

    var body: some View {
        NavigationStack {
            Group {
                if let at = position {
                    page(items[at], at: at)
                } else {
                    ContentUnavailableView(TrashText.emptyTitle, systemImage: "trash")
                }
            }
            .navigationTitle("In the trash")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Close") { index = nil }
                        .accessibilityIdentifier("trash.preview.close")
                }
            }
            .confirmationDialog("Delete forever", isPresented: purgeAsked, titleVisibility: .visible,
                                presenting: askPurge) { item in
                Button("Delete forever", role: .destructive) { onPurge(item) { askPurge = item } }
                Button("Cancel", role: .cancel) {}
            } message: { item in
                Text(TrashText.purgeQuestion(item.filename))
            }
        }
        .onChange(of: items.count) { _, count in
            if count == 0 {
                index = nil
            } else if let i = index, i >= count {
                index = count - 1
            }
        }
    }

    private func page(_ item: API.ItemCard, at: Int) -> some View {
        VStack(spacing: 14) {
            TrashPreviewMedia(item: item)
                .id(item.uid)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .gesture(DragGesture(minimumDistance: 30).onEnded { drag in
                    if drag.translation.width < -60 { move(1) } else if drag.translation.width > 60 { move(-1) }
                })
            HStack(spacing: 8) {
                Button { move(-1) } label: {
                    Image(systemName: "chevron.left").frame(width: 44, height: 44)
                }
                .disabled(at == 0)
                .accessibilityLabel("Previous item")
                .accessibilityIdentifier("trash.preview.previous")
                VStack(spacing: 2) {
                    Text(item.filename)
                        .font(.headline)
                        .lineLimit(2)
                        .multilineTextAlignment(.center)
                    Text(TrashText.position(at, of: max(total, items.count)))
                        .font(.footnote.monospacedDigit())
                        .foregroundStyle(.secondary)
                    if let length = TrashText.videoLength(kind: item.kind.rawValue, seconds: item.durationS) {
                        Text(length)
                            .font(.footnote.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .accessibilityIdentifier("trash.preview.length")
                    }
                }
                .frame(maxWidth: .infinity)
                .accessibilityElement(children: .combine)
                Button { move(1) } label: {
                    Image(systemName: "chevron.right").frame(width: 44, height: 44)
                }
                .disabled(at >= items.count - 1)
                .accessibilityLabel("Next item")
                .accessibilityIdentifier("trash.preview.next")
            }
            HStack(spacing: 12) {
                Button { onRestore(item) } label: {
                    Label("Restore", systemImage: "arrow.uturn.backward").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .accessibilityLabel("Restore \(item.filename)")
                .accessibilityIdentifier("trash.preview.restore")
                Button(role: .destructive) { askPurge = item } label: {
                    Label("Delete forever", systemImage: "trash").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .accessibilityLabel("Delete \(item.filename) forever")
                .accessibilityIdentifier("trash.preview.purge")
            }
            .controlSize(.large)
            .disabled(busy(item.uid))
        }
        .padding(16)
        .onAppear { onNearEnd(item) }
        // the next item's picture is fetched ahead, so moving on shows it at once
        .task(id: item.uid) { await prefetch(after: at) }
    }

    private func prefetch(after at: Int) async {
        let next = at + 1
        guard items.indices.contains(next), let picture = TrashPreviewMedia.picture(of: items[next]) else { return }
        _ = try? await ImageLoader.shared.image(picture.ref, maxPixel: picture.maxPixel)
    }

    private var purgeAsked: Binding<Bool> {
        Binding(get: { askPurge != nil }, set: { if !$0 { askPurge = nil } })
    }

    private func move(_ step: Int) {
        guard let at = position else { return }
        let to = at + step
        guard items.indices.contains(to) else { return }
        index = to
        onNearEnd(items[to])
    }
}

/// The big picture of the larger look.
struct TrashPreviewMedia: View {
    let item: API.ItemCard

    private var kind: MediaKind { MediaKind(item.kind.rawValue) }

    /// The picture the larger look loads for an item, and at what size (a video's poster, a document's first page,
    /// a photo's preview); nil for a private item, which shows the locked tile. The look fetches the next one ahead
    /// with the same address and size, so it is in memory when the owner moves on.
    static func picture(of item: API.ItemCard) -> (ref: String, maxPixel: Int)? {
        if item.nsfw { return nil }
        let kind = MediaKind(item.kind.rawValue)
        if kind == .video, item.urls.video != nil { return (item.urls.previewSmall, TrashVideo.posterPixels) }
        if kind == .pdf || kind == .document { return (item.urls.pages.first ?? item.urls.preview, 1600) }
        return (item.urls.preview, 1600)
    }

    var body: some View {
        if item.nsfw {
            TrashThumb(item: item, ref: item.urls.thumb)
        } else if kind == .video, let video = item.urls.video {
            TrashVideo(ref: video, posterRef: item.urls.previewSmall)
        } else if kind == .pdf || kind == .document {
            ZStack {
                Color.white
                LibraryKindImage(ref: item.urls.pages.first ?? item.urls.preview, maxPixel: 1600, contentMode: .fit,
                                 kind: kind)
                    .padding(8)
            }
            .accessibilityHidden(true)
        } else {
            ZStack {
                Color.black
                // a picture that cannot be loaded shows its kind's icon (the website's TrashPreview)
                LibraryKindImage(ref: item.urls.preview, maxPixel: 1600, contentMode: .fit, kind: kind)
            }
            .accessibilityHidden(true)
        }
    }
}

/// A trashed video with the system's controls; it starts only when asked.
struct TrashVideo: View {
    let ref: String
    let posterRef: String

    /// The poster's size (the look's prefetch asks for the same).
    static let posterPixels = 1280

    @State private var player: AVPlayer?
    /// The video cannot be played: its kind's icon shows instead (the website's TrashPreview).
    @State private var failed = false

    var body: some View {
        ZStack {
            Color.black
            if failed {
                Image(systemName: LibraryKindImage.symbol(.video))
                    .font(.largeTitle)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("This video cannot be played")
                    .accessibilityIdentifier("trash.preview.videoFailed")
            } else if let player {
                VideoPlayer(player: player)
            } else {
                LibraryKindImage(ref: posterRef, maxPixel: Self.posterPixels, contentMode: .fit, kind: .video)
            }
        }
        .onAppear {
            if player == nil && !failed {
                player = AuthorizedPlayer.player(for: ref)
                if player == nil { failed = true }
            }
        }
        .task(id: player.map { ObjectIdentifier($0) }) { await watch() }
        .onDisappear {
            player?.pause()
            player = nil
        }
    }

    /// Notices when the player gives up on the video.
    private func watch() async {
        guard let item = player?.currentItem else { return }
        for await status in item.publisher(for: \.status).values {
            if status == .failed {
                player?.pause()
                player = nil
                failed = true
                return
            }
        }
    }
}
