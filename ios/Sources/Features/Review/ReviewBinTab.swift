import SwiftUI

/// Mis-clicks or forwards (the website's MisclicksTab and ForwardsTab): the items with the reasons they were set
/// apart, selecting, Keep (back to the library) and Delete (to the trash, after a confirmation).
struct ReviewBinTab: View {
    enum Bin {
        case misclicks, forwards
    }

    let bin: Bin

    @Environment(AppRouter.self) private var router
    @State private var pager: Pager<API.ItemCard>
    @State private var selection = SelectionModel<String>()
    @State private var confirmDelete = false
    @State private var working = false

    init(bin: Bin) {
        self.bin = bin
        _pager = State(initialValue: Pager<API.ItemCard> { cursor in
            switch bin {
            case .misclicks:
                return try await Cloud.api().reviewMisclicks(cursor: cursor, limit: 60).pagerPage
            case .forwards:
                return try await Cloud.api().reviewForwards(cursor: cursor, limit: 60).pagerPage
            }
        })
    }

    var body: some View {
        ReviewItemGrid(items: pager.items, selection: selection, reasons: reasons,
                       onOpen: { item in open(item) },
                       onAppear: { item in pager.loadMoreIfNeeded(item) }) {
            VStack(alignment: .leading, spacing: 12) {
                Text(explanation)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if !pager.items.isEmpty && !selection.active {
                    Button(pager.finished ? "Select all" : "Select all loaded") {
                        selection.begin()
                        selection.select(pager.items.map(\.uid))
                    }
                    .buttonStyle(.bordered)
                    .accessibilityIdentifier("review.selectAll")
                }
                ReviewListState(pager: pager, emptyTitle: emptyTitle, emptyText: emptyText)
            }
            .padding(16)
        }
        .refreshable { await pager.refresh() }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if !selection.active {
                    Button("Select") { selection.begin() }
                        .disabled(pager.items.isEmpty)
                        .accessibilityIdentifier("review.select")
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            if selection.active {
                BulkBar(count: selection.count, actions: actions,
                        onSelectAll: { selection.select(pager.items.map(\.uid)) },
                        onDone: { selection.end() })
            }
        }
        .confirmAction("Move \(Fmt.count(selection.count, "item")) to the trash?", isPresented: $confirmDelete,
                       actionTitle: "Move to trash", message: "They can be restored from Trash until it is emptied.") {
            Task { await run(.delete) }
        }
        .task { if !pager.loaded { await pager.refresh() } }
        .onCloudChange { Task { await pager.refresh() } }
    }

    private var actions: [BulkAction] {
        [
            BulkAction("Keep", systemImage: "checkmark.circle", disabled: working) {
                Task { await run(bin == .misclicks ? .keep : .unforward) }
            },
            BulkAction("Delete", systemImage: "trash", role: .destructive, disabled: working) {
                confirmDelete = true
            },
        ]
    }

    private func reasons(_ item: API.ItemCard) -> [String] {
        switch bin {
        case .misclicks: return ReviewItemReasons.misclick(item.misclickReasons)
        case .forwards: return ReviewItemReasons.forward(item.forwardReasons)
        }
    }

    private var explanation: String {
        switch bin {
        case .misclicks:
            return "Blurry, dark, blown-out, pocket and burst shots. Nothing here is deleted until you say so. Keep returns an item to the library."
        case .forwards:
            return "Memes, greetings, posters and text images that came through WhatsApp or another messaging app. A real photo received the same way stays in the library. Keep returns an item to the library and teaches the app; nothing is deleted until you say so."
        }
    }

    private var emptyTitle: String { bin == .misclicks ? "No mis-clicks to review" : "No forwards to review" }

    private var emptyText: String {
        bin == .misclicks
            ? "Shots that look accidental show up here after upload."
            : "Media received through a messaging app that looks made for sharing shows up here after upload."
    }

    private func run(_ action: API.Action) async {
        let uids = Array(selection.selected)
        guard !uids.isEmpty, !working else { return }
        working = true
        defer { working = false }
        let forwards = bin == .forwards
        do {
            try await ReviewDecisions.bulk(uids, action: action) { n in
                let what = Fmt.count(n, "item")
                switch action {
                case .delete: return "\(what) moved to the trash."
                default: return forwards ? "\(what) kept in the library." : "\(what) kept."
                }
            }
            selection.end()
            CloudEvents.post(items: uids)
            await pager.refresh()
        } catch {
            Toaster.shared.error(error, doing: "update the selection")
        }
    }

    private func open(_ item: API.ItemCard) {
        router.show(pager.items, start: item) {
            await pager.loadMore()
            return pager.items.map(\.viewerEntry)
        }
    }
}

/// A grid of square thumbnails with the reasons each item waits here on its tile, selecting with a long press or
/// Select (the website's MediaGrid with reasons). It scrolls by itself, with `header` at its top.
struct ReviewItemGrid<Header: View>: View {
    let items: [API.ItemCard]
    let selection: SelectionModel<String>
    let reasons: (API.ItemCard) -> [String]
    let onOpen: (API.ItemCard) -> Void
    var onAppear: ((API.ItemCard) -> Void)?
    @ViewBuilder var header: () -> Header

    var body: some View {
        ScrollView {
            header()
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 2), count: 3), spacing: 2) {
                ForEach(items, id: \.uid) { item in
                    cell(item)
                }
            }
            .padding(.bottom, 24)
        }
        .accessibilityIdentifier("review.grid")
    }

    private func cell(_ item: API.ItemCard) -> some View {
        let picked = selection.contains(item.uid)
        let selecting = selection.active
        let why = reasons(item)
        return Color.clear
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                RemoteImage(ref: item.urls.thumb, maxPixel: 384)
            }
            .clipped()
            .overlay(alignment: .bottomLeading) {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(Array(why.prefix(3).enumerated()), id: \.offset) { _, reason in
                        ReviewReasonChip(text: reason)
                    }
                    if why.count > 3 {
                        ReviewReasonChip(text: "+\(why.count - 3)")
                    }
                }
                .padding(4)
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
            .onAppear { onAppear?(item) }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(([item.accessibilitySummary] + why).joined(separator: ", "))
            .accessibilityAddTraits(picked ? [.isButton, .isSelected] : .isButton)
            .accessibilityIdentifier("review.cell")
    }
}

/// One reason on a tile ("Too dark", "Meme 0.81").
struct ReviewReasonChip: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 10, weight: .semibold))
            .lineLimit(1)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .foregroundStyle(.white)
            .background(Color.orange, in: Capsule())
    }
}
