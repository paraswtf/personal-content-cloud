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
        RemoteImage(ref: ref, maxPixel: 2048, contentMode: .fit, vault: context.vault)
            .background(Color.white)
            .shadow(radius: 12)
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
                        ForEach(strip, id: \.self) { index in
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
            // ask again every 20 seconds until the copy is there
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 20_000_000_000)
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
/// VideoChips). A chip shows its first moment: the player pauses there and the box is drawn over the frame until
/// it plays again. Its pencil hands the box to the boxes area's editor.
struct ViewerVideoStage: View {
    let context: ViewerPageContext
    let item: API.ItemDetail
    let environment: ViewerStageEnvironment
    let boxes: ViewerBoxesModel
    let state: ViewerPageState

    @State private var player: AVPlayer?
    @State private var shown: ViewerFormat.VideoTrack?
    @State private var expanded: Set<String> = []

    private var groups: [ViewerFormat.VideoGroup] { ViewerFormat.groupTracks(ViewerFormat.videoTracks(item)) }

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 56)
            ZStack {
                if let player {
                    VideoPlayer(player: player)
                } else {
                    RemoteImage(ref: item.urls.previewSmall, maxPixel: 1600, contentMode: .fit, vault: context.vault)
                }
                if let shown {
                    GeometryReader { geo in
                        let rect = fitted(in: geo.size)
                        let x = CGFloat(shown.box.x), y = CGFloat(shown.box.y)
                        let w = CGFloat(shown.box.w), h = CGFloat(shown.box.h)
                        RoundedRectangle(cornerRadius: 4)
                            .stroke(Color.yellow, lineWidth: 2)
                            .frame(width: w * rect.width, height: h * rect.height)
                            .position(x: rect.minX + (x + w / 2) * rect.width, y: rect.minY + (y + h / 2) * rect.height)
                    }
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
                }
            }
            if !groups.isEmpty { chips }
            Spacer(minLength: 64)
        }
        .onChange(of: environment.isCurrent, initial: true) { _, current in
            if current {
                if player == nil, let video = item.urls.video {
                    player = AuthorizedPlayer.player(for: video, vault: context.vault)
                }
                state.players[context.uid] = player
            } else {
                player?.pause()
            }
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
        if player == nil, let video = item.urls.video {
            player = AuthorizedPlayer.player(for: video, vault: context.vault)
            state.players[context.uid] = player
        }
        guard let player else { return }
        player.pause()
        let time = CMTime(seconds: track.time, preferredTimescale: 600)
        player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero) { _ in
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
        let t = players[uid]?.currentTime().seconds ?? 0
        return t.isFinite ? max(0, t) : 0
    }
}
