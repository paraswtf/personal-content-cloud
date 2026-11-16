import SwiftUI

/// How the grid groups its items under headings.
enum GridSections {
    /// One heading per month ("November 2026"), from when each item was taken.
    case month
    /// No headings.
    case none
}

/// A grid of square thumbnails, newest first, with month headings, multi-select and pinch to change how many
/// fit across. Used by the library, albums, a person's photos, search and the trash. It scrolls by itself: put
/// it as a screen's main content, not inside another ScrollView.
struct MediaGrid<Item: MediaItem, Header: View>: View {
    let items: [Item]
    var sections: GridSections = .month
    /// While `selection.active`, taps pick items; a long press starts selecting.
    var selection: SelectionModel<String>?
    /// A tap outside selection: open the item (usually `router.show(items, start: item)`).
    let onOpen: (Item) -> Void
    /// Called as each cell appears: pass it to `pager.loadMoreIfNeeded`.
    var onAppear: ((Item) -> Void)?
    /// Shown at the top of the scroll view (filters, a summary).
    @ViewBuilder var header: () -> Header

    @AppStorage("grid.columns") private var columns = 4

    var body: some View {
        ScrollView {
            header()
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 2), count: max(2, min(8, columns))),
                      spacing: 2, pinnedViews: sections == .month ? [.sectionHeaders] : []) {
                ForEach(groups, id: \.key) { group in
                    Section {
                        ForEach(group.items) { item in cell(item) }
                    } header: {
                        if sections == .month { monthHeader(group.key) }
                    }
                }
            }
            .padding(.bottom, 24)
        }
        .simultaneousGesture(
            MagnifyGesture().onEnded { value in
                if value.magnification > 1.2 { columns = max(2, columns - 1) }
                if value.magnification < 0.83 { columns = min(8, columns + 1) }
            }
        )
        .accessibilityIdentifier("media.grid")
    }

    private struct Group {
        let key: String
        let items: [Item]
    }

    /// Items in their order, cut where the month changes (or one group without headings).
    private var groups: [Group] {
        guard sections == .month else { return [Group(key: "", items: items)] }
        var out: [Group] = []
        var currentKey: String?
        var bucket: [Item] = []
        for item in items {
            let key = item.monthKey
            if key != currentKey, !bucket.isEmpty {
                out.append(Group(key: currentKey ?? "", items: bucket))
                bucket = []
            }
            currentKey = key
            bucket.append(item)
        }
        if !bucket.isEmpty { out.append(Group(key: currentKey ?? "", items: bucket)) }
        // The same month may come back after another (items sorted by a different date): keep keys unique.
        var seen: [String: Int] = [:]
        return out.map { group in
            let n = seen[group.key, default: 0]
            seen[group.key] = n + 1
            return n == 0 ? group : Group(key: "\(group.key)#\(n)", items: group.items)
        }
    }

    private func monthHeader(_ key: String) -> some View {
        let plain = String(key.split(separator: "#").first ?? "")
        return Text(plain.isEmpty ? "Undated" : Fmt.monthHeading(plain))
            .font(.headline)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.bar)
            .accessibilityAddTraits(.isHeader)
    }

    private func cell(_ item: Item) -> some View {
        let picked = selection?.contains(item.uid) ?? false
        let selecting = selection?.active ?? false
        return Color.clear
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                RemoteImage(ref: item.thumbRef, maxPixel: 384)
            }
            .clipped()
            .overlay(alignment: .bottomLeading) { badges(item) }
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
                if let selection, selection.active {
                    selection.toggle(item.uid)
                } else {
                    onOpen(item)
                }
            }
            .onLongPressGesture(minimumDuration: 0.35) {
                guard let selection else { return }
                if selection.active { selection.toggle(item.uid) } else { selection.begin(with: item.uid) }
            }
            .onAppear { onAppear?(item) }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(item.accessibilitySummary)
            .accessibilityAddTraits(picked ? [.isButton, .isSelected] : .isButton)
            .accessibilityIdentifier("media.cell")
    }

    @ViewBuilder
    private func badges(_ item: Item) -> some View {
        HStack(spacing: 4) {
            if item.isFavorite {
                Image(systemName: "heart.fill").font(.caption2)
            }
            if item.isLive {
                Image(systemName: "livephoto").font(.caption2)
            }
            Spacer(minLength: 0)
            if item.mediaKind == .video {
                Text(Fmt.duration(item.durationSeconds)).font(.caption2.monospacedDigit().weight(.semibold))
            } else if item.mediaKind == .pdf || item.mediaKind == .document {
                Image(systemName: "doc.text").font(.caption2)
            }
        }
        .foregroundStyle(.white)
        .shadow(color: .black.opacity(0.6), radius: 2)
        .padding(5)
    }
}

extension MediaGrid where Header == EmptyView {
    init(items: [Item], sections: GridSections = .month, selection: SelectionModel<String>? = nil,
         onOpen: @escaping (Item) -> Void, onAppear: ((Item) -> Void)? = nil) {
        self.init(items: items, sections: sections, selection: selection, onOpen: onOpen, onAppear: onAppear) {
            EmptyView()
        }
    }
}
