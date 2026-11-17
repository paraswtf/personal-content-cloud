import SwiftUI
import AVKit
import Combine
import Observation

/// The pages the viewer area draws itself through `ViewerHooks.stage`: a document's pages (the website's
/// DocumentStage), a file without a preview (FileCard), and videos (VideoStage with the chips of the people and pets
/// in it, or VideoPreparing while the streaming copy is made). Photos keep the core's zoomable stage with its boxes.
enum ViewerStages {
    @MainActor
    static func stage(for context: ViewerPageContext, _ environment: ViewerStageEnvironment,
                      boxes: ViewerBoxesModel, pages: ViewerPageState) -> AnyView? {
        guard let item = context.item else { return nil }
        switch MediaKind(item.kind.rawValue) {
        case .pdf, .document:
            return AnyView(ViewerDocumentStage(context: context, item: item, environment: environment, state: pages))
        case .other:
            return AnyView(ViewerFileCard(context: context, item: item))
        case .video:
            if item.urls.video == nil {
                return AnyView(ViewerVideoPreparing(context: context, item: item))
            }
            return AnyView(ViewerVideoStage(context: context, item: item, environment: environment, boxes: boxes,
                                            state: pages))
        case .image:
            return nil
        }
    }
}

/// A document's pages: the page on screen large, previous and next, and a strip of the pages (a window around the
/// current one with the first and the last). Without page renders yet, the first page's preview and why.
struct ViewerDocumentStage: View {
    let context: ViewerPageContext
    let item: API.ItemDetail
    let environment: ViewerStageEnvironment
    let state: ViewerPageState

    @State private var page = 0
    @State private var file: ViewerFile?
    @State private var opening = false

    private var pages: [String] { item.urls.pages }
    private var label: String {
        item.document.map { ViewerFormat.docTypeLabel($0.docType) } ?? item.filename
    }

    var body: some View {
        VStack(spacing: 8) {
            Spacer(minLength: 96)
            if pages.isEmpty {
                pageImage(item.urls.preview, label: "\(label), first page")
                HStack(spacing: 8) {
                    if item.textStatus == .pending { ProgressView().tint(.white) }
                    Text(item.textStatus == .pending ? "Pages are still being read." : "Only the first page has a preview.")
                }
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.7))
            } else {
                let current = min(max(page, 0), pages.count - 1)
                pageImage(pages[current], label: "\(label), page \(current + 1) of \(pages.count)")
                    .id(current)
                controls(current: current)
            }
            Spacer(minLength: 72)
        }
        .padding(.horizontal, 12)
        .contentShape(Rectangle())
        .onTapGesture { environment.toggleChrome() }
        .sheet(item: $file) { file in
            ViewerQuickLook(url: file.url).ignoresSafeArea().onDisappear { ViewerFiles.discard(file) }
        }
        .onChange(of: page, initial: true) { _, now in state.page[context.uid] = now }
    }

    private func pageImage(_ ref: String, label: String) -> some View {
        ViewerPageImage(ref: ref, vault: context.vault)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityElement()
            .accessibilityLabel(label)
            .accessibilityIdentifier("viewer.document.page")
    }

    private func controls(current: Int) -> some View {
        let strip = Self.stripPages(current: current, count: pages.count)
        return HStack(spacing: 8) {
            Button {
                page = current - 1
            } label: {
                Image(systemName: "chevron.left").frame(width: 40, height: 40)
            }
            .disabled(current <= 0)
            .accessibilityLabel("Previous page")
            .accessibilityIdentifier("viewer.document.previous")
            ScrollViewReader { proxy in
                ScrollView(.horizontal, showsIndicators: false) {
                    LazyHStack(spacing: 6) {
                        ForEach(Array(strip.enumerated()), id: \.element) { pos, index in
                            if Self.gapBefore(strip, at: pos) {
                                Text("...")
                                    .font(.footnote)
                                    .foregroundStyle(.white.opacity(0.6))
                                    .frame(width: 16)
                                    .accessibilityHidden(true)
                            }
                            Button {
                                page = index
                            } label: {
                                RemoteImage(ref: pages[index], maxPixel: 160, contentMode: .fill, vault: context.vault)
                                    .frame(width: 36, height: 48)
                                    .background(Color.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 3))
                                    .overlay(alignment: .bottom) {
                                        Text("\(index + 1)")
                                            .font(.system(size: 9, weight: .semibold))
                                            .frame(maxWidth: .infinity)
                                            .background(.black.opacity(0.55))
                                            .foregroundStyle(.white)
                                    }
                                    .overlay(RoundedRectangle(cornerRadius: 3)
                                        .stroke(index == current ? Color.accentColor : .clear, lineWidth: 2))
                                    .opacity(index == current ? 1 : 0.7)
                            }
                            .id(index)
                            .accessibilityLabel("Page \(index + 1)")
                            .accessibilityAddTraits(index == current ? .isSelected : [])
                        }
                    }
                }
                .onChange(of: current, initial: true) { _, now in
                    withAnimation { proxy.scrollTo(now, anchor: .center) }
                }
            }
            .frame(height: 52)
            Text("\(current + 1) / \(pages.count)")
                .font(.caption.monospacedDigit())
                .accessibilityLabel("Page \(current + 1) of \(pages.count)")
                .accessibilityIdentifier("viewer.document.position")
            Button {
                page = current + 1
            } label: {
                Image(systemName: "chevron.right").frame(width: 40, height: 40)
            }
            .disabled(current >= pages.count - 1)
            .accessibilityLabel("Next page")
            .accessibilityIdentifier("viewer.document.next")
            if MediaKind(item.kind.rawValue) == .pdf, let original = item.urls.original {
                Button {
                    Task { await open(original) }
                } label: {
                    if opening { ProgressView().tint(.white).frame(width: 40, height: 40) }
                    else { Image(systemName: "arrow.up.forward.square").frame(width: 40, height: 40) }
                }
                .disabled(opening)
                .accessibilityLabel("Open the PDF")
                .accessibilityIdentifier("viewer.document.open")
            }
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 4)
    }

    /// The pages the strip shows: a window of `span` pages either side of the current one, and the first and the
    /// last page when the window does not reach them.
    static func stripPages(current: Int, count: Int, span: Int = 30) -> [Int] {
        guard count > 0 else { return [] }
        let lo = max(0, current - span), hi = min(count - 1, current + span)
        var pages: [Int] = []
        if lo > 0 { pages.append(0) }
        pages.append(contentsOf: lo...hi)
        if hi < count - 1 { pages.append(count - 1) }
        return pages
    }

    /// Whether pages are left out of the strip just before its `pos`-th page (the website's "..." between the first
    /// or last page and the window).
    static func gapBefore(_ strip: [Int], at pos: Int) -> Bool {
        pos > 0 && pos < strip.count && strip[pos] > strip[pos - 1] + 1
    }

    private func open(_ original: String) async {
        opening = true
        defer { opening = false }
        do {
            file = try await ViewerFiles.fetch(context, address: original, filename: item.filename)
        } catch {
            Toaster.shared.error(error, doing: "open the PDF")
        }
    }
}

/// One page of a document: a spinner while it loads, and the website's words when it could not be loaded.
struct ViewerPageImage: View {
    let ref: String
    var vault = false

    @State private var image: UIImage?
    @State private var failed = false

    var body: some View {
        ZStack {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .background(Color.white)
                    .shadow(radius: 12)
            } else if failed {
                Text("This page could not be loaded.")
                    .font(.subheadline)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white.opacity(0.8))
                    .padding(16)
                    .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
            } else {
                ProgressView()
                    .tint(.white)
                    .controlSize(.large)
                    .accessibilityLabel("Loading page")
            }
        }
        .task(id: ref) { await load() }
    }

    private func load() async {
        failed = false
        if let hit = ImageLoader.shared.cached(ref, maxPixel: 2048, vault: vault) {
            image = hit
            return
        }
        image = nil
        do {
            image = try await ImageLoader.shared.image(ref, maxPixel: 2048, vault: vault)
        } catch {
            if error is CancellationError || Task.isCancelled { return }
            failed = true
        }
    }
}

/// A file the server has no preview for: its name, type and size, and the original to download.
struct ViewerFileCard: View {
    let context: ViewerPageContext
    let item: API.ItemDetail

    @State private var file: ViewerFile?
    @State private var busy = false

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "doc.text")
                .font(.system(size: 32))
                .frame(width: 64, height: 64)
                .background(.white.opacity(0.1), in: RoundedRectangle(cornerRadius: 16))
            VStack(spacing: 4) {
                Text(item.filename).font(.headline).multilineTextAlignment(.center)
                Text("\(ViewerFormat.extLabel(item.filename)) file, \(Fmt.bytes(item.sizeBytes))")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.6))
            }
            Text("There is no preview for this kind of file. The original is kept exactly as uploaded.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.7))
            if let download = item.urls.download {
                Button {
                    Task {
                        busy = true
                        defer { busy = false }
                        do { file = try await ViewerFiles.fetch(context, address: download, filename: item.filename) }
                        catch { Toaster.shared.error(error, doing: "download the original") }
                    }
                } label: {
                    HStack {
                        Label("Download original", systemImage: "arrow.down.circle")
                        if busy { ProgressView() }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(busy)
                .accessibilityIdentifier("viewer.file.download")
            }
        }
        .foregroundStyle(.white)
        .padding(24)
        .background(.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 20))
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .sheet(item: $file) { file in
            ViewerShareSheet(items: [file.url]).onDisappear { ViewerFiles.discard(file) }
        }
    }
}

/// Shown instead of the player while the server makes the compressed streaming copy; the viewer reads the item again
/// now and then and the player takes over once it is there.
struct ViewerVideoPreparing: View {
    let context: ViewerPageContext
    let item: API.ItemDetail

    @State private var file: ViewerFile?
    @State private var busy = false

    var body: some View {
        ZStack {
            RemoteImage(ref: item.urls.previewSmall, maxPixel: 1280, contentMode: .fit, vault: context.vault)
            Color.black.opacity(0.55)
            VStack(spacing: 12) {
                ProgressView().tint(.white).controlSize(.large)
                Text("Preparing a lighter version for smooth playback.")
                    .font(.subheadline.weight(.semibold))
                    .accessibilityIdentifier("viewer.video.preparing")
                Text("This usually takes a minute.").font(.subheadline).foregroundStyle(.white.opacity(0.75))
                if let download = item.urls.download {
                    Button {
                        Task {
                            busy = true
                            defer { busy = false }
                            do { file = try await ViewerFiles.fetch(context, address: download, filename: item.filename) }
                            catch { Toaster.shared.error(error, doing: "download the original") }
                        }
                    } label: {
                        Label("Download original (\(Fmt.bytes(item.sizeBytes)))", systemImage: "arrow.down.circle")
                            .font(.subheadline)
                    }
                    .disabled(busy)
                }
            }
            .multilineTextAlignment(.center)
            .foregroundStyle(.white)
            .padding(24)
        }
        .task(id: context.uid) {
            // ask again every 5 seconds until the copy is there, as the website does
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                if Task.isCancelled { break }
                context.reload()
            }
        }
        .sheet(item: $file) { file in
            ViewerShareSheet(items: [file.url]).onDisappear { ViewerFiles.discard(file) }
        }
    }
}

/// A video with the people and pets in it: the player, and below it one chip per person or look-alike (the website's
/// VideoChips). A chip shows its first moment: the player pauses there and the box is drawn over the frame, with the
/// name and who put it there, until it plays again. Its pencil hands the box to the boxes area's editor. While a box
/// is drawn or moved on this video, the paused frame shows as a picture the boxes area's stage edits (the website's
/// VideoStage draw and adjust modes). The video opens paused at `context.startAt` (a search hit's moment).
struct ViewerVideoStage: View {
    let context: ViewerPageContext
    let item: API.ItemDetail
    let environment: ViewerStageEnvironment
    let boxes: ViewerBoxesModel
    let state: ViewerPageState

    @State private var player: AVPlayer?
    @State private var shown: ViewerFormat.VideoTrack?
    @State private var expanded: Set<String> = []
    /// The paused frame being drawn on, and its time.
    @State private var frameImage: UIImage?
    @State private var frameImageTime: Double?
    @State private var frameZoomed = false

    private var tracks: [ViewerFormat.VideoTrack] { ViewerFormat.videoTracks(item) }
    private var groups: [ViewerFormat.VideoGroup] { ViewerFormat.groupTracks(tracks) }
    /// A box is being drawn or moved on this video.
    private var editingHere: Bool { boxes.stage.isEditing(context.uid) }

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 56)
            ZStack {
                if editingHere {
                    frameStage
                } else {
                    if let player {
                        VideoPlayer(player: player)
                    } else {
                        RemoteImage(ref: item.urls.previewSmall, maxPixel: 1600, contentMode: .fit, vault: context.vault)
                    }
                    if let shown, environment.showBoxes {
                        shownBox(shown)
                    }
                }
            }
            if !groups.isEmpty && environment.showBoxes && !editingHere { chips }
            Spacer(minLength: 64)
        }
        .onChange(of: environment.isCurrent, initial: true) { _, current in
            if current {
                state.players[context.uid] = makePlayer()
            } else {
                player?.pause()
            }
        }
        .onChange(of: editingHere) { _, now in
            if now {
                // the frame under the box stays put while it is drawn or moved
                player?.pause()
                shown = nil
            } else {
                frameImage = nil
                frameImageTime = nil
                frameZoomed = false
            }
        }
        .task(id: "\(context.uid)#\(editingHere)#\(boxes.stage.frameTime ?? -1)") { await loadFrame() }
        .onChange(of: boxes.videoFocus) { _, focus in
            guard let focus, focus.uid == context.uid else { return }
            boxes.videoFocus = nil
            seek(to: focus.time, showing: tracks.first(where: { $0.id == focus.boxKey }))
        }
        .onReceive(NotificationCenter.default.publisher(for: AVPlayerItem.timeJumpedNotification)) { note in
            // the owner moved the player elsewhere: the chip's box no longer matches the frame
            if let changed = note.object as? AVPlayerItem, changed === player?.currentItem, shown != nil,
               let now = player?.currentTime().seconds, let at = shown?.time, abs(now - at) > 0.3 {
                shown = nil
            }
        }
        .onReceive(playing) { status in
            // the box belongs to the paused frame: it goes once the video plays
            if status == .playing { shown = nil }
        }
        .onDisappear {
            player?.pause()
            player = nil
            state.players[context.uid] = nil
        }
    }

    /// The player, made once the page is on screen, paused at the entry's moment when it has one.
    @discardableResult
    private func makePlayer() -> AVPlayer? {
        if let player { return player }
        guard let video = item.urls.video else { return nil }
        let made = AuthorizedPlayer.player(for: video, vault: context.vault)
        if let startAt = context.startAt, startAt > 0 {
            made?.seek(to: CMTime(seconds: startAt, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
        }
        player = made
        state.players[context.uid] = made
        return made
    }

    /// The paused frame as a picture, with the boxes area's drawing or moving layer over it.
    private var frameStage: some View {
        let size: CGSize? = {
            guard let w = item.width, let h = item.height, w > 0, h > 0 else { return nil }
            return CGSize(width: w, height: h)
        }()
        return ZStack {
            ZoomableStage(image: frameImage,
                          pixelSize: size,
                          boxes: [],
                          showBoxes: true,
                          selectedBoxId: nil,
                          zoomed: $frameZoomed,
                          onBoxTap: { _ in },
                          onTap: {},
                          edit: boxes.stage,
                          editRevision: boxes.stage.revision)
            if frameImage == nil {
                ProgressView().tint(.white).accessibilityLabel("Loading the frame")
            }
        }
    }

    /// Loads the frame being drawn on: the server's picture of it (the one it crops the new box from), else the
    /// player's own, else the video's poster.
    private func loadFrame() async {
        guard editingHere, let t = boxes.stage.frameTime else { return }
        if frameImage != nil && frameImageTime == t { return }
        frameImage = nil
        let rounded = (t * 1000).rounded() / 1000
        let version = ViewerActions.fileVersion(item)
        let uid = context.uid
        var data: Data?
        do {
            data = try await APIContext.$vault.withValue(context.vault) {
                try await Cloud.api().frame(uid: uid, t: rounded, v: version, size: "preview")
            }
        } catch {
            data = nil
        }
        if let data, let image = UIImage(data: data) {
            guard !Task.isCancelled else { return }
            frameImage = image
            frameImageTime = t
            return
        }
        guard !Task.isCancelled else { return }
        if let asset = player?.currentItem?.asset {
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.requestedTimeToleranceBefore = .zero
            generator.requestedTimeToleranceAfter = .zero
            if let cg = try? await generator.image(at: CMTime(seconds: t, preferredTimescale: 600)).image {
                guard !Task.isCancelled else { return }
                frameImage = UIImage(cgImage: cg)
                frameImageTime = t
                return
            }
        }
        guard !Task.isCancelled else { return }
        frameImage = try? await ImageLoader.shared.image(item.urls.previewSmall, maxPixel: 1280, vault: context.vault)
        frameImageTime = t
    }

    /// The chip's box over the paused frame, with its name and who put it there (the website's label).
    private func shownBox(_ shown: ViewerFormat.VideoTrack) -> some View {
        GeometryReader { geo in
            let rect = fitted(in: geo.size)
            let box = CGRect(x: rect.minX + CGFloat(shown.box.x) * rect.width,
                             y: rect.minY + CGFloat(shown.box.y) * rect.height,
                             width: CGFloat(shown.box.w) * rect.width, height: CGFloat(shown.box.h) * rect.height)
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 4)
                    .stroke(Color.yellow, lineWidth: 2)
                    .frame(width: box.width, height: box.height)
                    .offset(x: box.minX, y: box.minY)
                HStack(spacing: 4) {
                    Text(shown.name).lineLimit(1)
                    if let origin = shown.origin { OriginIcon(origin: origin, decorative: true) }
                }
                .font(.caption.weight(.semibold))
                .foregroundStyle(.black)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.yellow, in: RoundedRectangle(cornerRadius: 3))
                .frame(maxWidth: 200, alignment: .leading)
                .fixedSize()
                .offset(x: box.minX - 1, y: box.minY >= 24 ? box.minY - 22 : box.maxY + 2)
            }
            .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    /// The player's playing or paused state, as it changes.
    private var playing: AnyPublisher<AVPlayer.TimeControlStatus, Never> {
        guard let player else { return Empty().eraseToAnyPublisher() }
        return player.publisher(for: \.timeControlStatus).removeDuplicates().eraseToAnyPublisher()
    }

    /// Where the video's picture sits inside the player (fitted, centred).
    private func fitted(in size: CGSize) -> CGRect {
        let w = CGFloat(item.width ?? 16), h = CGFloat(item.height ?? 9)
        guard w > 0, h > 0, size.width > 0, size.height > 0 else { return CGRect(origin: .zero, size: size) }
        let scale = min(size.width / w, size.height / h)
        let fit = CGSize(width: w * scale, height: h * scale)
        return CGRect(x: (size.width - fit.width) / 2, y: (size.height - fit.height) / 2, width: fit.width, height: fit.height)
    }

    private var chips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(groups) { group in
                    let whole = group.tracks.count > 1
                    chip(group.first, count: group.tracks.count, groupKey: whole ? group.key : nil, origin: whole ? group.origin : group.first.origin)
                    if whole && expanded.contains(group.key) {
                        ForEach(group.tracks) { track in
                            chip(track, count: 1, groupKey: nil, origin: track.origin, inner: true)
                        }
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .accessibilityLabel("People and pets in this video")
        .accessibilityIdentifier("viewer.video.chips")
    }

    private func chip(_ track: ViewerFormat.VideoTrack, count: Int, groupKey: String?, origin: BoxOrigin?,
                      inner: Bool = false) -> some View {
        let whole = count > 1 && groupKey != nil
        let how = origin.map { ", \($0.label)" } ?? ""
        let active = shown?.id == track.id
        return HStack(spacing: 0) {
            Button {
                seek(to: track)
            } label: {
                HStack(spacing: 6) {
                    RemoteImage(ref: track.crop, maxPixel: 96, vault: context.vault)
                        .frame(width: 30, height: 30)
                        .clipShape(Circle())
                    if track.box.kind == .animal { Image(systemName: "pawprint").imageScale(.small) }
                    Text(track.name)
                        .lineLimit(1)
                        .foregroundStyle(track.isGuess ? Color.yellow : Color.white)
                    if let origin { OriginIcon(origin: origin, decorative: true).foregroundStyle(.white.opacity(0.7)) }
                    Text(whole ? "\(count) tracks" : Fmt.duration(track.time))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.6))
                }
                .padding(.leading, 4)
                .padding(.trailing, 10)
                .frame(height: 40)
            }
            .accessibilityLabel((whole ? "\(track.name), \(count) tracks, show the first" : "\(track.name), \(Fmt.duration(track.time)), show this frame") + how)
            Button {
                boxes.tap(track.box, in: context)
            } label: {
                Image(systemName: "pencil").frame(width: 36, height: 40)
            }
            .accessibilityLabel(whole ? "Edit all \(count) tracks of \(track.name)"
                                : track.isNamed ? "Edit, tagged as \(track.name)"
                                : track.box.kind == .animal ? "Name this pet" : "Tag this person")
            if let groupKey, whole {
                Button {
                    if expanded.contains(groupKey) { expanded.remove(groupKey) } else { expanded.insert(groupKey) }
                } label: {
                    Image(systemName: expanded.contains(groupKey) ? "chevron.up" : "chevron.down").frame(width: 32, height: 40)
                }
                .accessibilityLabel(expanded.contains(groupKey) ? "Hide the tracks of \(track.name)" : "Show the \(count) tracks of \(track.name)")
            }
        }
        .font(.subheadline)
        .foregroundStyle(.white)
        .background((active ? Color.white.opacity(0.25) : Color.white.opacity(0.1)), in: Capsule())
        .overlay(Capsule().strokeBorder(track.isNamed ? Color.clear : Color.yellow.opacity(0.7),
                                        style: StrokeStyle(lineWidth: 1, dash: track.isGuess ? [] : [3, 3])))
        .scaleEffect(inner ? 0.95 : 1)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("viewer.video.chip")
    }

    private func seek(to track: ViewerFormat.VideoTrack) {
        seek(to: track.time, showing: track)
    }

    /// Pauses the player at `time` and draws `track`'s box once it is there.
    private func seek(to time: Double, showing track: ViewerFormat.VideoTrack?) {
        guard let player = makePlayer() else { return }
        player.pause()
        shown = nil
        player.seek(to: CMTime(seconds: time, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero) { _ in
            Task { @MainActor in shown = track }
        }
    }
}

/// What the viewer's own pages share with its toolbar: the document page on screen and the video player of each item,
/// so the downloads can offer "this page" and "this frame" as the website's menu does.
@MainActor
@Observable
final class ViewerPageState {
    var page: [String: Int] = [:]
    @ObservationIgnored var players: [String: AVPlayer] = [:]

    /// The time of the frame on screen in seconds (0 before the video played).
    func frameTime(_ uid: String) -> Double {
        playerTime(uid) ?? 0
    }

    /// The time of the frame on screen in seconds, or nil when the item has no player yet.
    func playerTime(_ uid: String) -> Double? {
        guard let player = players[uid] else { return nil }
        let t = player.currentTime().seconds
        return t.isFinite ? max(0, t) : 0
    }
}
