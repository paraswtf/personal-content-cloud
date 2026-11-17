import SwiftUI

// The library's grids (the website's components/timeline/TimelineGrid.tsx and Scrubber.tsx, and the search side of
// components/MediaGrid.tsx): the timeline by month with a scrubber, the search results with their snippets, and the
// tile both use.

/// The column count both grids share with the core's media grid (pinch to change it).
private let columnsKey = "grid.columns"

private func gridColumns(_ n: Int) -> [GridItem] {
    Array(repeating: GridItem(.flexible(), spacing: 2), count: max(2, min(8, n)))
}

/// The timeline: every month of the skeleton with its heading and its tiles, loaded page by page as they come on
/// screen (tiles not loaded yet are grey squares of the final size, so the scrubber lands where it should).
struct LibraryTimelineView: View {
    let model: LibraryModel
    let vault: Bool
    let header: AnyView
    let onOpen: (API.ItemCard) -> Void

    @AppStorage(columnsKey) private var columns = 4
    @State private var scrubbing: String?
    /// Where the list is scrolled to, for the scrubber's mark (only the mark reads it, so scrolling redraws nothing
    /// else).
    @State private var position = LibraryScrollPosition()

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                header
                // the month's heading stays at the top while its tiles are on screen (the website's sticky header)
                LazyVStack(alignment: .leading, spacing: 0, pinnedViews: [.sectionHeaders]) {
                    ForEach(model.buckets, id: \.key) { bucket in
                        LibraryMonthSection(model: model, bucket: bucket, columns: columns, vault: vault, onOpen: onOpen)
                    }
                }
                // the months' frame in the scroll view: the scrubber's mark follows it (the website measures the grid)
                .background(GeometryReader { geo in
                    Color.clear.preference(key: LibraryScrollFrameKey.self,
                                           value: geo.frame(in: .named(LibraryScrollPosition.space)))
                })
                .padding(.bottom, 24)
            }
            .coordinateSpace(name: LibraryScrollPosition.space)
            .background(GeometryReader { geo in
                Color.clear
                    .onAppear { position.viewport = geo.size.height }
                    .onChange(of: geo.size.height) { _, height in position.viewport = height }
            })
            .onPreferenceChange(LibraryScrollFrameKey.self) { frame in position.update(frame) }
            .accessibilityIdentifier("library.timeline")
            .overlay(alignment: .trailing) {
                if model.buckets.count > 3 {
                    LibraryScrubber(buckets: model.buckets, position: position, scrubbing: $scrubbing) { key in
                        proxy.scrollTo(key, anchor: .top)
                    }
                }
            }
            .overlay(alignment: .top) {
                if let scrubbing {
                    Text(scrubbing)
                        .font(.headline)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(.regularMaterial, in: Capsule())
                        .padding(.top, 12)
                        .accessibilityHidden(true)
                }
            }
        }
        .simultaneousGesture(
            MagnifyGesture().onEnded { value in
                if value.magnification > 1.2 { columns = max(2, columns - 1) }
                if value.magnification < 0.83 { columns = min(8, columns + 1) }
            }
        )
    }
}

/// One month: its heading ("March 2024", the count, Select month while selecting) and its tiles.
struct LibraryMonthSection: View {
    let model: LibraryModel
    let bucket: API.TimelineBucket
    let columns: Int
    let vault: Bool
    let onOpen: (API.ItemCard) -> Void

    var body: some View {
        let month = model.months[bucket.key]
        let cards = month?.cards ?? []
        let missing = month?.complete == true ? 0 : max(0, bucket.count - cards.count)
        let label = Fmt.monthHeading(bucket.key)
        Section {
            tiles(cards: cards, missing: missing)
        } header: {
            HStack(spacing: 8) {
                (Text(label).font(.headline)
                    + Text("  \(Fmt.count(bucket.count, "item"))").font(.subheadline).foregroundColor(.secondary))
                    .accessibilityAddTraits(.isHeader)
                Spacer(minLength: 8)
                if model.selection.active && !cards.isEmpty {
                    let uids = cards.map(\.uid)
                    let all = uids.allSatisfy { model.selection.contains($0) }
                    Button(all ? "Deselect month" : "Select month") {
                        if all { model.selection.deselect(uids) } else { model.selection.select(uids) }
                    }
                    .font(.subheadline.weight(.semibold))
                    .accessibilityLabel("\(all ? "Deselect" : "Select") month \(label)")
                }
            }
            .padding(.horizontal, 12)
            .padding(.top, 14)
            .padding(.bottom, 8)
            .background(Color(uiColor: .systemBackground))
        }
    }

    private func tiles(cards: [API.ItemCard], missing: Int) -> some View {
        let month = model.months[bucket.key]
        return VStack(alignment: .leading, spacing: 0) {
            LazyVGrid(columns: gridColumns(columns), spacing: 2) {
                ForEach(cards) { card in
                    LibraryTile(item: card, vault: vault, selection: model.selection, onOpen: onOpen)
                }
                ForEach(0..<missing, id: \.self) { i in
                    LibraryPlaceholderTile()
                        .onAppear { model.ensureMonth(bucket.key, need: cards.count + i + 1) }
                }
            }
            if month?.failed == true {
                HStack(spacing: 12) {
                    Text("Could not load these items.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Button("Try again") { model.retryMonth(bucket.key) }
                        .buttonStyle(.bordered)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            }
        }
        // on screen, and again whenever the months start over (a month that was on screen during a load asks then)
        .task(id: model.monthsToken) { model.ensureMonth(bucket.key) }
    }
}

/// The search results in one grid, with the matched words of each and the matched moment of a video.
struct LibrarySearchResults: View {
    let model: LibraryModel
    let vault: Bool
    let header: AnyView
    let onOpen: (API.SearchHit) -> Void

    @AppStorage(columnsKey) private var columns = 4

    var body: some View {
        ScrollView {
            header
            LazyVGrid(columns: gridColumns(columns), spacing: 2) {
                ForEach(model.hits) { hit in
                    LibraryTile(item: hit, vault: vault, selection: model.selection, showSnippet: true, onOpen: onOpen)
                        .onAppear {
                            if model.hits.suffix(12).contains(where: { $0.uid == hit.uid }) {
                                Task { await model.loadMoreHits() }
                            }
                        }
                }
            }
            if model.loadingMoreHits {
                ProgressView().padding()
            }
            if model.moreHitsFailed {
                VStack(spacing: 8) {
                    Text("Could not load more.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Button("Try again") { Task { await model.retryMoreHits() } }
                        .buttonStyle(.bordered)
                }
                .padding(.vertical, 16)
            }
            Spacer(minLength: 24)
        }
        .simultaneousGesture(
            MagnifyGesture().onEnded { value in
                if value.magnification > 1.2 { columns = max(2, columns - 1) }
                if value.magnification < 0.83 { columns = min(8, columns + 1) }
            }
        )
        .accessibilityIdentifier("library.results")
    }
}

/// A square tile: the thumb, the website's badges (favourite, Live, mis-click, forward, similar copies, a
/// document's type, a video's length), and while selecting a check mark. A tap opens the item or, while selecting,
/// picks it; a long press starts selecting.
struct LibraryTile<Item: LibraryTileItem>: View {
    let item: Item
    let vault: Bool
    let selection: SelectionModel<String>
    var showSnippet = false
    let onOpen: (Item) -> Void

    var body: some View {
        let picked = selection.contains(item.uid)
        let selecting = selection.active
        let snippet = showSnippet ? item.snippetText.flatMap { $0.isEmpty ? nil : $0 } : nil
        Color.clear
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                if item.isDocumentTile {
                    documentPage
                } else {
                    RemoteImage(ref: item.thumbRef, maxPixel: 384, vault: vault)
                }
            }
            .clipped()
            .overlay(alignment: .topLeading) { flags }
            .overlay(alignment: .bottomTrailing) {
                if item.mediaKind == .video {
                    badge {
                        Image(systemName: "play.fill")
                        if let d = item.durationSeconds, d > 0 {
                            Text(Fmt.duration(d)).monospacedDigit()
                        } else {
                            Text("Video")
                        }
                    }
                    .padding(4)
                }
            }
            .overlay(alignment: .bottomLeading) {
                if let at = item.matchSeconds {
                    badge(tint: .accentColor) { Text("at \(Fmt.duration(at))").monospacedDigit() }
                        .padding(4)
                } else if snippet == nil && item.isDocumentTile {
                    badge(tint: Color.black.opacity(0.75)) { Text(documentLabel).lineLimit(1) }
                        .padding(4)
                }
            }
            .overlay(alignment: .bottom) {
                if let snippet {
                    Text(LibrarySnippet.attributed(snippet))
                        .font(.caption2)
                        .foregroundStyle(.white)
                        .lineLimit(3)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 5)
                        .padding(.top, 14)
                        .padding(.bottom, 4)
                        .background(LinearGradient(colors: [.clear, .black.opacity(0.85)], startPoint: .top,
                                                   endPoint: .bottom))
                }
            }
            .overlay(alignment: .topTrailing) {
                if selecting {
                    Image(systemName: picked ? "checkmark.circle.fill" : "circle")
                        .font(.title3)
                        .foregroundStyle(picked ? Color.accentColor : Color.white)
                        .shadow(radius: 2)
                        .padding(5)
                }
            }
            .overlay {
                if picked { Rectangle().stroke(Color.accentColor, lineWidth: 3) }
            }
            .contentShape(Rectangle())
            .onTapGesture {
                if selection.active {
                    selection.toggle(item.uid)
                } else {
                    onOpen(item)
                }
            }
            .onLongPressGesture(minimumDuration: 0.35) {
                if selection.active { selection.toggle(item.uid) } else { selection.begin(with: item.uid) }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(snippet.map { "\(item.libraryLabel), \(LibrarySnippet.plain($0))" } ?? item.libraryLabel)
            .accessibilityAddTraits(picked ? [.isButton, .isSelected] : .isButton)
            .accessibilityIdentifier("media.cell")
    }

    /// A document's first page on a sheet of paper, fitted from the top (the website's inset white page), with the
    /// document icon when it cannot be loaded.
    private var documentPage: some View {
        GeometryReader { geo in
            let inset = min(geo.size.width, geo.size.height) * 0.1
            LibraryKindImage(ref: item.thumbRef, maxPixel: 384, contentMode: .fit, vault: vault,
                             kind: item.mediaKind == .image ? .document : item.mediaKind, alignment: .top)
                .background(Color.white)
                .clipShape(RoundedRectangle(cornerRadius: 3))
                .shadow(color: .black.opacity(0.25), radius: 1.5, y: 1)
                .scaleEffect(selection.contains(item.uid) ? 0.9 : 1)
                .padding(inset)
        }
        .background(Color(uiColor: .secondarySystemFill))
    }

    private var documentLabel: String {
        let base = item.docType != nil ? LibraryDocTypes.label(item.docType) : LibraryDocTypes.extensionLabel(item.filename)
        if let pages = item.pageCount, pages > 1 { return "\(base) (\(pages) p)" }
        return base
    }

    @ViewBuilder
    private var flags: some View {
        HStack(spacing: 3) {
            if item.isFavorite {
                Image(systemName: "star.fill")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.yellow)
                    .padding(4)
                    .background(Color.black.opacity(0.55), in: Circle())
            }
            if item.isLive {
                badge { Image(systemName: "livephoto"); Text("LIVE").fontWeight(.bold) }
            }
            if item.misclick {
                badge(tint: .orange) { Image(systemName: "exclamationmark.triangle.fill") }
            }
            if item.forward {
                badge { Image(systemName: "arrowshape.turn.up.right.fill") }
            }
            if item.dupCount > 0 {
                badge { Image(systemName: "square.on.square"); Text("+\(item.dupCount)") }
            }
        }
        .padding(4)
    }

    private func badge<Content: View>(tint: Color = Color.black.opacity(0.6),
                                      @ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 2) { content() }
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(tint, in: Capsule())
    }
}

/// A tile whose item is not loaded yet.
struct LibraryPlaceholderTile: View {
    var body: some View {
        Color(uiColor: .secondarySystemFill)
            .aspectRatio(1, contentMode: .fit)
            .accessibilityHidden(true)
    }
}

/// The scrubber (the website's Scrubber.tsx): a thin rail on the right with the years at their share of the
/// timeline; dragging or tapping goes to the month under the finger and names it.
struct LibraryScrubber: View {
    let buckets: [API.TimelineBucket]
    let position: LibraryScrollPosition
    @Binding var scrubbing: String?
    let jump: (String) -> Void

    @State private var lastKey: String?
    @State private var focus = 0

    private struct YearMark: Identifiable {
        let year: String
        let fraction: Double
        var id: String { year }
    }

    var body: some View {
        let total = max(1, buckets.reduce(0) { $0 + $1.count })
        GeometryReader { geo in
            let height = max(1, geo.size.height)
            ZStack(alignment: .topTrailing) {
                Capsule()
                    .fill(Color.secondary.opacity(0.35))
                    .frame(width: 3)
                    .frame(maxHeight: .infinity)
                    .padding(.trailing, 4)
                LibraryScrubberMark(position: position, width: geo.size.width, height: height)
                ForEach(yearMarks(total: total)) { mark in
                    Text(mark.year)
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 3)
                        .background(.thinMaterial, in: Capsule())
                        .position(x: geo.size.width / 2 - 2, y: min(height - 6, max(6, CGFloat(mark.fraction) * height)))
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let i = index(at: Double(value.location.y / height), total: total)
                        let key = buckets[i].key
                        scrubbing = Fmt.monthHeading(key)
                        focus = i
                        if key != lastKey {
                            lastKey = key
                            jump(key)
                        }
                    }
                    .onEnded { _ in
                        scrubbing = nil
                        lastKey = nil
                    }
            )
        }
        .frame(width: 26)
        .padding(.vertical, 12)
        .accessibilityElement()
        .accessibilityLabel("Jump to a month")
        .accessibilityValue(buckets.isEmpty ? "" : Fmt.monthHeading(buckets[min(focus, buckets.count - 1)].key))
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: focus = min(buckets.count - 1, focus + 1)
            case .decrement: focus = max(0, focus - 1)
            @unknown default: break
            }
            if buckets.indices.contains(focus) { jump(buckets[focus].key) }
        }
        .accessibilityIdentifier("library.scrubber")
    }

    /// The month at a share of the rail's height, by the months' counts.
    private func index(at fraction: Double, total: Int) -> Int {
        let target = min(max(fraction, 0), 0.9999) * Double(total)
        var seen = 0
        for (i, bucket) in buckets.enumerated() {
            seen += bucket.count
            if Double(seen) > target { return i }
        }
        return max(0, buckets.count - 1)
    }

    /// Where each year starts on the rail (only when there is more than one year).
    private func yearMarks(total: Int) -> [YearMark] {
        var marks: [YearMark] = []
        var seen = 0
        var lastYear = ""
        for bucket in buckets {
            let year = String(bucket.key.prefix(4))
            if year != lastYear {
                marks.append(YearMark(year: year, fraction: Double(seen) / Double(total)))
                lastYear = year
            }
            seen += bucket.count
        }
        return marks.count > 1 ? marks : []
    }
}

/// How far down the timeline is scrolled (0 at the top, 1 at the end), for the scrubber's mark.
@MainActor
@Observable
final class LibraryScrollPosition {
    static let space = "library.timeline.scroll"

    private(set) var fraction: Double = 0
    /// The scroll view's height.
    @ObservationIgnored var viewport: CGFloat = 0

    /// The months' frame in the scroll view: its top moves up as the list scrolls.
    func update(_ content: CGRect) {
        let next = LibraryScrollPosition.fraction(contentTop: content.minY, contentHeight: content.height,
                                                  viewport: viewport)
        if abs(next - fraction) > 0.001 { fraction = next }
    }

    nonisolated static func fraction(contentTop: CGFloat, contentHeight: CGFloat, viewport: CGFloat) -> Double {
        let room = contentHeight - viewport
        guard room > 1 else { return 0 }
        return min(1, max(0, Double(-contentTop / room)))
    }
}

/// The timeline content's frame in its scroll view.
struct LibraryScrollFrameKey: PreferenceKey {
    static var defaultValue: CGRect = .zero
    static func reduce(value: inout CGRect, nextValue: () -> CGRect) { value = nextValue() }
}

/// The accent line on the scrubber's rail at the scroll position (the website's Scrubber mark).
struct LibraryScrubberMark: View {
    let position: LibraryScrollPosition
    let width: CGFloat
    let height: CGFloat

    var body: some View {
        Capsule()
            .fill(Color.accentColor)
            .frame(width: 12, height: 2.5)
            .position(x: width - 5.5, y: min(height - 1, max(1, CGFloat(position.fraction) * height)))
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }
}
