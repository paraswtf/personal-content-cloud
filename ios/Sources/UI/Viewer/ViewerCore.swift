import SwiftUI
import AVKit
import Combine

/// The item on screen in the viewer: the core's view of it (what the stage draws), the whole generated
/// `API.ItemDetail` (nil only if it could not be decoded; `json` holds the raw answer), and a way to load it
/// again after a change. `vault` is true when the viewer was opened from the vault: wrap the item's calls in
/// `APIContext.$vault.withValue(context.vault) { ... }` and pass `vault: context.vault` to pictures and players.
struct ViewerPageContext: Identifiable {
    let uid: String
    let detail: CoreItemDetail
    let item: API.ItemDetail?
    let json: Data
    let reload: () -> Void
    var vault = false
    /// A video opens paused at this second (a search hit's matching moment, a face's frame): the entry's `startAt`.
    var startAt: Double? = nil
    /// Opened in document mode (the Documents page): the info panel opens at once and puts the document first.
    var documentFirst = false
    /// Takes this item out of the viewer's list and shows the next one (the one before at the end), or closes the
    /// viewer when it was the last: after the item went to the trash or the vault, as the website goes on.
    var removeFromViewer: () -> Void = {}

    var id: String { uid }
}

/// What a page the viewer area draws itself (`ViewerHooks.stage`) gets besides the item.
struct ViewerStageEnvironment {
    /// The page on screen: play only then.
    let isCurrent: Bool
    /// Shows or hides the viewer's bars, as a single tap on the core's picture does.
    let toggleChrome: () -> Void
    /// Tells the viewer the page is zoomed in, so a swipe pans the page instead of going to the next one.
    let setZoomed: (Bool) -> Void
    /// The bottom bar's Show or Hide boxes: a video hides its people chips and the box on its frame too.
    var showBoxes = true
}

/// What the viewer area adds to the core viewer (Features/Viewer/ViewerScreen.swift fills these).
struct ViewerHooks {
    /// A tapped box (the smallest under the finger): open the face, person or pet editor.
    var onBoxTap: ((StageBox, ViewerPageContext) -> Void)?
    /// The info panel's content for the item on screen.
    var infoPanel: ((ViewerPageContext) -> AnyView)?
    /// Extra buttons for the bottom bar (favourite, share, delete, albums).
    var toolbar: ((ViewerPageContext) -> AnyView)?
    /// A page the viewer area draws itself (a document's pages, a note, its own video player). Return nil to keep
    /// the core's picture, video or Live Photo for that item.
    var stage: ((ViewerPageContext, ViewerStageEnvironment) -> AnyView?)?
    /// The box being edited, drawn selected (never raised above a smaller one).
    var selectedBoxId: String?

    init(onBoxTap: ((StageBox, ViewerPageContext) -> Void)? = nil,
         infoPanel: ((ViewerPageContext) -> AnyView)? = nil,
         toolbar: ((ViewerPageContext) -> AnyView)? = nil,
         stage: ((ViewerPageContext, ViewerStageEnvironment) -> AnyView?)? = nil,
         selectedBoxId: String? = nil) {
        self.onBoxTap = onBoxTap
        self.infoPanel = infoPanel
        self.toolbar = toolbar
        self.stage = stage
        self.selectedBoxId = selectedBoxId
    }
}

/// The viewer: pages through items left and right, zooms photos, plays videos and Live Photo clips, and draws the
/// boxes of faces, people and pets with the smallest on top.
struct ViewerCore: View {
    let request: ViewerRequest
    var hooks: ViewerHooks

    @Environment(\.dismiss) private var dismiss
    @State private var entries: [ViewerEntry]
    @State private var current: String?
    @State private var chrome = true
    @State private var showBoxes = true
    @State private var zoomedUid: String?
    @State private var contexts: [String: ViewerPageContext] = [:]
    @State private var info: ViewerPageContext?
    @State private var loadingMore = false
    @State private var noMore = false
    /// Document mode opened the info panel once already.
    @State private var openedDocumentPanel = false
    /// The whole list's size when the screen that opened the viewer knows it; one less for each item taken out.
    @State private var total: Int?

    init(request: ViewerRequest, hooks: ViewerHooks = ViewerHooks()) {
        self.request = request
        self.hooks = hooks
        _entries = State(initialValue: request.entries)
        _total = State(initialValue: request.total)
        let start = request.entries.indices.contains(request.start) ? request.entries[request.start].uid : request.entries.first?.uid
        _current = State(initialValue: start)
    }

    private var currentContext: ViewerPageContext? { current.flatMap { contexts[$0] } }
    private var currentIndex: Int? { current.flatMap { uid in entries.firstIndex(where: { $0.uid == uid }) } }
    /// The boxes area's edit (a box moved or drawn), when the boxes area put one in the environment.
    @Environment(StageEditState.self) private var edit: StageEditState?
    /// A box is being moved or drawn: the bars hide, paging stops and the info panel closes until it ends.
    private var editing: Bool { edit?.isActive ?? false }

    var body: some View {
        ScrollViewReader { proxy in
            pages
                .onAppear {
                    // Start at the opened item (the scroll position alone may not move a lazy stack on its first layout).
                    if let current { proxy.scrollTo(current, anchor: .leading) }
                }
        }
        .background(Color.black.ignoresSafeArea())
        .overlay(alignment: .top) { if chrome && !editing { topBar } }
        .overlay(alignment: .bottom) { if chrome && !editing { bottomBar } }
        .statusBarHidden(!chrome)
        .preferredColorScheme(.dark)
        .onChange(of: current) { _, _ in
            Task { await loadMoreIfNeeded() }
        }
        .onChange(of: editing) { _, nowEditing in
            if nowEditing { info = nil }
        }
        .onChange(of: edit?.showBoxesRequest ?? 0) { _, _ in
            // the boxes area shows a person or a box, or starts moving or drawing one: the boxes come back on
            showBoxes = true
        }
        .sheet(item: $info) { opened in
            // The item as it is now: a save in the panel reloads the page, and the panel shows the new answer.
            let ctx = contexts[opened.uid] ?? opened
            NavigationStack {
                Group {
                    if let panel = hooks.infoPanel {
                        panel(ctx)
                    } else {
                        DefaultInfoPanel(context: ctx)
                    }
                }
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) { Button("Done") { info = nil } }
                }
            }
            .presentationDetents([.medium, .large])
        }
    }

    private var pages: some View {
        ScrollView(.horizontal) {
            LazyHStack(spacing: 0) {
                ForEach(entries) { entry in
                    ViewerPage(entry: entry,
                               isCurrent: entry.uid == current,
                               showBoxes: showBoxes,
                               selectedBoxId: hooks.selectedBoxId,
                               zoomed: zoomBinding(entry.uid),
                               onTap: { withAnimation(.easeInOut(duration: 0.2)) { chrome.toggle() } },
                               onBoxTap: { box, ctx in hooks.onBoxTap?(box, ctx) },
                               onContext: { ctx in
                                   contexts[ctx.uid] = ctx
                                   edit?.contexts[ctx.uid] = ctx
                                   // Document mode: the panel opens with the item the owner tapped.
                                   if request.mode == .document, !openedDocumentPanel, ctx.uid == startUid {
                                       openedDocumentPanel = true
                                       info = ctx
                                   }
                               },
                               stage: hooks.stage,
                               vault: request.vault,
                               documentFirst: request.mode == .document,
                               onRemove: { uid in remove(uid) })
                        .containerRelativeFrame([.horizontal, .vertical])
                        .id(entry.uid)
                }
            }
            .scrollTargetLayout()
        }
        .scrollTargetBehavior(.paging)
        .scrollPosition(id: $current)
        .scrollIndicators(.hidden)
        .scrollDisabled(zoomedUid != nil || editing)
        .ignoresSafeArea()
        // On the scroll view itself: on the whole viewer it would replace the identifiers of its buttons.
        .accessibilityIdentifier("viewer")
    }

    private func zoomBinding(_ uid: String) -> Binding<Bool> {
        Binding(get: { zoomedUid == uid },
                set: { isZoomed in
                    if isZoomed { zoomedUid = uid } else if zoomedUid == uid { zoomedUid = nil }
                })
    }

    private var topBar: some View {
        HStack(spacing: 12) {
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 17, weight: .semibold))
                    .frame(width: 44, height: 44)
            }
            .accessibilityLabel("Close")
            .accessibilityIdentifier("viewer.close")
            VStack(alignment: .leading, spacing: 2) {
                if let ctx = currentContext {
                    Text(Fmt.date(ctx.detail.takenAt ?? ctx.detail.createdAt, withTime: true))
                        .font(.subheadline.weight(.semibold))
                    if let place = ctx.detail.place, !place.isEmpty {
                        Text(place).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .lineLimit(1)
            Spacer()
            if let counter, counter.shown {
                Text(counter.text)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .padding(.trailing, 12)
                    .accessibilityLabel(counter.spoken)
                    .accessibilityIdentifier("viewer.counter")
            }
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 4)
        .background(LinearGradient(colors: [.black.opacity(0.65), .clear], startPoint: .top, endPoint: .bottom)
            .ignoresSafeArea())
    }

    private var bottomBar: some View {
        HStack(spacing: 20) {
            if let ctx = currentContext {
                if !ctx.detail.stillBoxes.isEmpty || Self.videoHasBoxes(ctx) {
                    Button {
                        showBoxes.toggle()
                    } label: {
                        Image(systemName: showBoxes ? "person.crop.rectangle.fill" : "person.crop.rectangle")
                            .frame(width: 44, height: 44)
                    }
                    .accessibilityLabel(showBoxes ? "Hide boxes" : "Show boxes")
                    .accessibilityIdentifier("viewer.boxes")
                }
                if let extra = hooks.toolbar {
                    extra(ctx)
                }
                Spacer()
                Button {
                    info = ctx
                } label: {
                    Image(systemName: "info.circle").frame(width: 44, height: 44)
                }
                .accessibilityLabel("Details")
                .accessibilityIdentifier("viewer.info")
            } else {
                Spacer()
            }
        }
        .font(.system(size: 20))
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(LinearGradient(colors: [.clear, .black.opacity(0.65)], startPoint: .top, endPoint: .bottom)
            .ignoresSafeArea())
    }

    /// A video with people or pets on its frames: the toggle hides its chips (the website's boxes button).
    static func videoHasBoxes(_ ctx: ViewerPageContext) -> Bool {
        guard ctx.detail.mediaKind == .video, let item = ctx.item else { return false }
        return item.faces.contains { $0.frameTimeS != nil } || item.bodies.contains { $0.frameTimeS != nil }
            || item.animals.contains { $0.frameTimeS != nil }
    }

    /// The top bar's counter for the item on screen.
    private var counter: ViewerCounter? {
        guard let current, let i = currentIndex else { return nil }
        return ViewerCounter.make(index: i, place: request.positionOf?(current), loaded: entries.count, total: total,
                                  moreMayCome: !(noMore || request.loadMore == nil))
    }

    /// The item the viewer was opened on.
    private var startUid: String? {
        request.entries.indices.contains(request.start) ? request.entries[request.start].uid : request.entries.first?.uid
    }

    /// Takes an item out of the list and shows its neighbour (the next one, else the one before); closes the viewer
    /// when none is left.
    private func remove(_ uid: String) {
        guard let i = entries.firstIndex(where: { $0.uid == uid }) else { return }
        info = nil
        guard entries.count > 1 else {
            dismiss()
            return
        }
        let neighbour = i + 1 < entries.count ? entries[i + 1].uid : entries[i - 1].uid
        if current == uid { current = neighbour }
        entries.remove(at: i)
        contexts[uid] = nil
        if let t = total { total = max(0, t - 1) }
    }

    private func loadMoreIfNeeded() async {
        guard let more = request.loadMore, !loadingMore, !noMore, let i = currentIndex, i >= entries.count - 3 else { return }
        loadingMore = true
        let next = await more()
        loadingMore = false
        let known = Set(entries.map(\.uid))
        let fresh = next.filter { !known.contains($0.uid) }
        if fresh.isEmpty { noMore = true } else { entries.append(contentsOf: fresh) }
    }
}

/// The viewer's counter (the website's Viewer position): the item's place in the whole list and the list's size when
/// the screen that opened the viewer knows them ("3 of 120"), else its place among the items loaded and how many are
/// loaded, with a "+" while paging on can load more ("3 of 40+").
struct ViewerCounter: Equatable {
    let number: Int
    let of: Int
    let more: Bool

    /// `index`: the item's place in the viewer's entries (from 0). `place`: its place in the whole list (from 0),
    /// when the entries are only parts of it. `loaded`: the entries. `total`: the whole list's size, when known.
    /// `moreMayCome`: paging past the last entry can load more.
    static func make(index: Int, place: Int?, loaded: Int, total: Int?, moreMayCome: Bool) -> ViewerCounter {
        guard let total else { return ViewerCounter(number: index + 1, of: loaded, more: moreMayCome) }
        let at = place ?? index
        // as the website: never fewer than the items loaded, nor than the item's own place
        return ViewerCounter(number: at + 1, of: max(total, loaded, at + 1), more: false)
    }

    /// Shown once there is more than one item.
    var shown: Bool { of > 1 }

    var text: String { "\(number.formatted()) of \(of.formatted())\(more ? "+" : "")" }

    /// What VoiceOver reads (the website's "Item 3 of 120").
    var spoken: String { more ? "Item \(number) of at least \(of)" : "Item \(number) of \(of)" }
}

/// One page of the viewer: the item's picture (zoomable, with boxes), its video, or its Live Photo clip.
struct ViewerPage: View {
    let entry: ViewerEntry
    let isCurrent: Bool
    let showBoxes: Bool
    let selectedBoxId: String?
    @Binding var zoomed: Bool
    let onTap: () -> Void
    let onBoxTap: (StageBox, ViewerPageContext) -> Void
    let onContext: (ViewerPageContext) -> Void
    /// The viewer area's own page for an item (ViewerHooks.stage).
    var stage: ((ViewerPageContext, ViewerStageEnvironment) -> AnyView?)? = nil
    /// Opened from the vault: the item, its pictures and its videos are asked for with `vault=1`.
    var vault = false
    /// Document mode (ViewerRequest.mode): handed to the context for the info panel.
    var documentFirst = false
    /// Takes an item out of the viewer's list (ViewerPageContext.removeFromViewer).
    var onRemove: (String) -> Void = { _ in }

    @Environment(StageEditState.self) private var edit: StageEditState?
    @State private var detail: CoreItemDetail?
    @State private var item: API.ItemDetail?
    @State private var json: Data?
    @State private var image: UIImage?
    @State private var loadError: Error?
    @State private var reloadToken = 0
    /// Bumped after each poll, so the next one waits its turn (a poll reads the item again without reloading the
    /// picture).
    @State private var pollRound = 0
    @State private var livePlayer: AVPlayer?
    /// The large preview could not be loaded (the thumb stays, with the website's note).
    @State private var previewFailed = false

    private var kind: MediaKind { MediaKind(detail?.kind ?? entry.kind) }

    var body: some View {
        let custom = customStage
        ZStack {
            if let custom {
                custom
            } else if kind == .video, let detail, let video = detail.urls.video {
                ViewerVideo(ref: video, posterRef: detail.urls.previewSmall, isCurrent: isCurrent, vault: vault,
                            startAt: entry.startAt)
            } else {
                let editingHere = edit?.isEditing(entry.uid) ?? false
                ZoomableStage(image: image,
                              pixelSize: pixelSize,
                              boxes: detail?.stillBoxes ?? [],
                              // the boxes show while one of them is moved or a new one drawn
                              showBoxes: editingHere || (showBoxes && livePlayer == nil),
                              selectedBoxId: selectedBoxId,
                              zoomed: $zoomed,
                              onBoxTap: { box in if let ctx = context { onBoxTap(box, ctx) } },
                              onTap: onTap,
                              edit: editingHere ? edit : nil,
                              editRevision: editingHere ? (edit?.revision ?? 0) : -1)
                if let livePlayer {
                    // A tap on the playing clip stops it, as on the website.
                    PlayerSurface(player: livePlayer)
                        .contentShape(Rectangle())
                        .onTapGesture { stopLive() }
                        .accessibilityHidden(true)
                }
            }
            if previewFailed && detail != nil && custom == nil && kind != .video {
                VStack {
                    Spacer()
                    Text("The preview could not be loaded. You can still download the original.")
                        .font(.footnote)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 12))
                        .padding(.horizontal, 24)
                        .padding(.bottom, 96)
                        .accessibilityIdentifier("viewer.previewFailed")
                }
                .allowsHitTesting(false)
            }
            if detail == nil, let loadError {
                VStack(spacing: 12) {
                    Text(ErrorText.describe(loadError)).multilineTextAlignment(.center)
                    Button("Try again") { reloadToken += 1 }
                        .buttonStyle(.bordered)
                }
                .foregroundStyle(.white)
                .padding(24)
            } else if image == nil && kind != .video && custom == nil {
                ProgressView().tint(.white)
            }
        }
        .overlay(alignment: .topLeading) {
            if let liveRef, kind == .image, custom == nil {
                Button {
                    if livePlayer == nil { playLive(liveRef) } else { stopLive() }
                } label: {
                    Label("LIVE", systemImage: livePlayer == nil ? "livephoto" : "livephoto.slash")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                .foregroundStyle(.white)
                .padding(.top, 104)
                .padding(.leading, 12)
                .accessibilityLabel(livePlayer == nil ? "Play the Live Photo" : "Stop the Live Photo")
                .accessibilityIdentifier("viewer.live")
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(accessibilityText)
        .task(id: "\(entry.uid)#\(reloadToken)") { await load() }
        .task(id: pollKey) { await poll() }
        .onCloudChange(uid: entry.uid) { reloadToken += 1 }
        .onReceive(NotificationCenter.default.publisher(for: .AVPlayerItemDidPlayToEndTime)) { note in
            // the clip loops, muted, until it is tapped or the owner leaves the item (the website's LiveButton)
            if let item = note.object as? AVPlayerItem, item === livePlayer?.currentItem {
                livePlayer?.seek(to: .zero)
                livePlayer?.play()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .AVPlayerItemFailedToPlayToEndTime)) { note in
            if let item = note.object as? AVPlayerItem, item === livePlayer?.currentItem { liveFailed() }
        }
        .onReceive(liveStatus) { status in
            if status == .failed { liveFailed() }
        }
        .onChange(of: isCurrent) { _, nowCurrent in
            if !nowCurrent { stopLive() }
        }
        .onChange(of: edit?.showBoxesRequest ?? 0) { _, _ in
            // boxes are not drawn over a playing Live Photo clip (the website's showPerson, showBox, startAdjust)
            stopLive()
        }
        .onChange(of: edit?.isEditing(entry.uid) ?? false) { _, nowEditing in
            if nowEditing { stopLive() }
        }
    }

    private var pixelSize: CGSize? {
        guard let w = detail?.width, let h = detail?.height, w > 0, h > 0 else { return nil }
        return CGSize(width: w, height: h)
    }

    private var liveRef: String? {
        guard let detail, detail.live == true else { return nil }
        return detail.urls.live ?? detail.liveVideo?.url
    }

    private var context: ViewerPageContext? {
        guard let detail, let json else { return nil }
        let uid = entry.uid
        let remove = onRemove
        return ViewerPageContext(uid: uid, detail: detail, item: item, json: json, reload: { reloadToken += 1 },
                                 vault: vault, startAt: entry.startAt, documentFirst: documentFirst,
                                 removeFromViewer: { remove(uid) })
    }

    /// The viewer area's own page for this item, when it draws one.
    private var customStage: AnyView? {
        guard let stage, let ctx = context else { return nil }
        let zoom = $zoomed
        return stage(ctx, ViewerStageEnvironment(isCurrent: isCurrent, toggleChrome: onTap,
                                                 setZoomed: { zoom.wrappedValue = $0 }, showBoxes: showBoxes))
    }

    private var accessibilityText: String {
        let date = Fmt.date(detail?.takenAt ?? detail?.createdAt, withTime: true)
        return [kind.label, date].filter { !$0.isEmpty }.joined(separator: ", ")
    }

    private func playLive(_ ref: String) {
        guard let player = AuthorizedPlayer.player(for: ref, vault: vault) else {
            Toaster.shared.error("The clip could not be played")
            return
        }
        player.isMuted = true
        livePlayer = player
        player.play()
    }

    private func stopLive() {
        livePlayer?.pause()
        livePlayer = nil
    }

    private func liveFailed() {
        guard livePlayer != nil else { return }
        stopLive()
        Toaster.shared.error("The clip could not be played")
    }

    /// The Live Photo clip's loading state, as it changes (a clip that cannot load is reported).
    private var liveStatus: AnyPublisher<AVPlayerItem.Status, Never> {
        guard let item = livePlayer?.currentItem else { return Empty().eraseToAnyPublisher() }
        return item.publisher(for: \.status).removeDuplicates().eraseToAnyPublisher()
    }

    /// What the page waits for while the server still works on the item: the website reads it again every 4
    /// seconds while faces or text are pending, and every 5 seconds while a video's new copy is made (a video
    /// without any copy yet asks again from its own page). Only the page on screen asks.
    private var pollSeconds: UInt64? {
        guard isCurrent, let item else { return nil }
        if item.analysisStatus == .pending || item.textStatus == .pending { return 4 }
        if MediaKind(item.kind.rawValue) == .video && item.previewStatus == .pending && item.urls.video != nil { return 5 }
        return nil
    }

    private var pollKey: String { "\(entry.uid)#\(pollSeconds ?? 0)#\(pollRound)" }

    /// Reads the item again, as the website's refetch does: the details change, the picture being loaded (the large
    /// preview on a slow link) goes on loading.
    private func poll() async {
        guard let seconds = pollSeconds else { return }
        try? await Task.sleep(nanoseconds: seconds * 1_000_000_000)
        if Task.isCancelled { return }
        await refreshDetail()
        if Task.isCancelled { return }
        pollRound += 1
    }

    /// The item's details read again without starting the picture over. Only a new version of the picture (its
    /// preview's address changed: rotated, edited) loads the whole page again.
    private func refreshDetail() async {
        guard let server = APIConnection.current else { return }
        let query = vault ? [URLQueryItem(name: "vault", value: "1")] : []
        guard let answer = try? await CoreHTTP.raw(server, "GET", "/api/items/\(entry.uid)", query: query),
              let decoded = try? CoreHTTP.decoder.decode(CoreItemDetail.self, from: answer.0) else { return }
        let data = answer.0
        if Task.isCancelled { return }
        if decoded.mediaKind != .video, let shown = detail?.urls.preview, shown != decoded.urls.preview {
            reloadToken += 1
            return
        }
        detail = decoded
        item = try? JSONDecoder().decode(API.ItemDetail.self, from: data)
        json = data
        if let ctx = context { onContext(ctx) }
    }

    private func load() async {
        if image == nil, let thumb = entry.thumb {
            image = ImageLoader.shared.cached(thumb, maxPixel: 512, vault: vault)
            if image == nil { image = try? await ImageLoader.shared.image(thumb, maxPixel: 512, vault: vault) }
        }
        guard let server = APIConnection.current else { return }
        do {
            let query = vault ? [URLQueryItem(name: "vault", value: "1")] : []
            let (data, _) = try await CoreHTTP.raw(server, "GET", "/api/items/\(entry.uid)", query: query)
            let decoded = try CoreHTTP.decoder.decode(CoreItemDetail.self, from: data)
            detail = decoded
            item = try? JSONDecoder().decode(API.ItemDetail.self, from: data)
            json = data
            loadError = nil
            if let ctx = context { onContext(ctx) }
            if image == nil {
                image = try? await ImageLoader.shared.image(decoded.urls.thumb, maxPixel: 512, vault: vault)
            }
            if decoded.mediaKind != .video {
                do {
                    let preview = try await ImageLoader.shared.image(decoded.urls.preview, maxPixel: 2560, vault: vault)
                    image = preview
                    previewFailed = false
                } catch {
                    if error is CancellationError || Task.isCancelled { return }
                    previewFailed = true
                }
            }
        } catch {
            if error is CancellationError || Task.isCancelled { return }
            if detail == nil { loadError = error }
        }
    }
}

/// A video page: the poster until the page is the current one, then the system player (it does not start by itself).
struct ViewerVideo: View {
    let ref: String
    let posterRef: String
    let isCurrent: Bool
    var vault = false
    /// Opens paused at this second (ViewerEntry.startAt).
    var startAt: Double? = nil

    @State private var player: AVPlayer?

    var body: some View {
        ZStack {
            if let player {
                VideoPlayer(player: player)
            } else {
                RemoteImage(ref: posterRef, maxPixel: 1600, contentMode: .fit, vault: vault)
            }
        }
        .onChange(of: isCurrent, initial: true) { _, nowCurrent in
            if nowCurrent {
                if player == nil {
                    player = AuthorizedPlayer.player(for: ref, vault: vault)
                    if let startAt, startAt > 0 {
                        player?.seek(to: CMTime(seconds: startAt, preferredTimescale: 600),
                                     toleranceBefore: .zero, toleranceAfter: .zero)
                    }
                }
            } else {
                player?.pause()
            }
        }
        .onDisappear {
            player?.pause()
            player = nil
        }
    }
}

/// The info panel when the viewer area has not given one: what the item is and when it was taken.
struct DefaultInfoPanel: View {
    let context: ViewerPageContext

    var body: some View {
        let d = context.detail
        List {
            LabeledContent("Name", value: d.filename)
            LabeledContent("Kind", value: d.mediaKind.label)
            if let taken = d.takenAt { LabeledContent("Taken", value: Fmt.date(taken, withTime: true)) }
            LabeledContent("Added", value: Fmt.date(d.createdAt, withTime: true))
            let size = Fmt.resolution(d.width, d.height)
            if !size.isEmpty { LabeledContent("Size", value: size) }
            if let place = d.place, !place.isEmpty { LabeledContent("Place", value: place) }
        }
        .navigationTitle("Details")
        .navigationBarTitleDisplayMode(.inline)
    }
}
