import SwiftUI

/// One group of near-identical pictures: the best copy (the one the library shows) and the others.
struct ReviewDupGroup: Identifiable {
    let group: API.DupGroup

    var id: String { group.best.uid }
    var all: [API.ItemCard] { [group.best] + group.copies }
}

/// Duplicates (the website's DuplicatesTab): each group with its best copy, Make best on another copy, and Delete
/// copy (to the trash, after a confirmation).
struct ReviewDuplicatesTab: View {
    @Environment(AppRouter.self) private var router
    @State private var pager = Pager<ReviewDupGroup> { cursor in
        let page = try await Cloud.api().duplicates(cursor: cursor, limit: 30)
        return Pager<ReviewDupGroup>.Page(items: page.groups.map(ReviewDupGroup.init), next: page.nextCursor,
                                          total: page.total)
    }
    @State private var deleting: API.ItemCard?
    @State private var working: String?
    /// The running action is a delete (the tapped Make best shows a spinner otherwise).
    @State private var deletingNow = false

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                Text("Near-identical pictures are grouped. The library shows only the best copy; the others stay here, kept at full quality, until you delete them.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                ForEach(pager.items) { group in
                    groupCard(group)
                        .onAppear { pager.loadMoreIfNeeded(group) }
                }
                ReviewListState(pager: pager, emptyTitle: "No duplicates",
                                emptyText: "Similar copies of the same picture show up here.",
                                emptyImage: "square.on.square")
            }
            .padding(16)
        }
        .refreshable { await pager.refresh() }
        .confirmationDialog("Delete this copy?", isPresented: Binding(get: { deleting != nil },
                                                                      set: { if !$0 { deleting = nil } }),
                            titleVisibility: .visible, presenting: deleting) { copy in
            Button("Move to trash", role: .destructive) { Task { await delete(copy) } }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text("It moves to the trash. The best copy stays in the library.")
        }
        .task { if !pager.loaded { await pager.refresh() } }
        .onCloudChange { Task { await pager.refresh() } }
    }

    private func groupCard(_ group: ReviewDupGroup) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(Fmt.count(group.all.count, "copy", "copies"))
                .font(.subheadline.weight(.semibold))
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 12) {
                    tile(group.group.best, best: true, group: group)
                    ForEach(group.group.copies, id: \.uid) { copy in
                        tile(copy, best: false, group: group)
                    }
                }
            }
        }
        .padding(12)
        .background(Color(uiColor: .secondarySystemGroupedBackground),
                    in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
            .stroke(Color(uiColor: .separator), lineWidth: 0.5))
    }

    private func tile(_ item: API.ItemCard, best: Bool, group: ReviewDupGroup) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                open(item)
            } label: {
                Color.clear
                    .aspectRatio(1, contentMode: .fit)
                    .overlay { RemoteImage(ref: item.urls.thumb, maxPixel: 384) }
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(best ? Color.green : Color(uiColor: .separator), lineWidth: best ? 2 : 0.5))
                    .overlay(alignment: .topLeading) {
                        if best {
                            Label("Best", systemImage: "crown.fill")
                                .font(.system(size: 11, weight: .semibold))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .foregroundStyle(.white)
                                .background(Color.green, in: Capsule())
                                .padding(5)
                        }
                    }
            }
            .buttonStyle(.plain)
            .accessibilityLabel(best ? "Open the best copy, \(item.filename)" : "Open this copy, \(item.filename)")
            let size = Fmt.resolution(item.width, item.height)
            Text(size.isEmpty ? item.filename : size)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Text(Fmt.date(item.takenAt ?? item.createdAt))
                .font(.caption)
                .foregroundStyle(.tertiary)
            if !best {
                Button {
                    Task { await makeBest(item) }
                } label: {
                    HStack(spacing: 4) {
                        if working == item.uid && !deletingNow {
                            ProgressView().controlSize(.small)
                        }
                        Text("Make best")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(working != nil)
                .accessibilityLabel("Make best, \(item.filename)")
                .accessibilityIdentifier("review.dup.best")
                Button(role: .destructive) {
                    deleting = item
                } label: {
                    Text("Delete copy").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(working != nil)
                .accessibilityLabel("Delete copy, \(item.filename)")
                .accessibilityIdentifier("review.dup.delete")
            }
        }
        .frame(width: 140)
    }

    /// Every copy of every group loaded, in order, for the viewer (the website's flat list).
    private static func flat(_ groups: [ReviewDupGroup]) -> [API.ItemCard] {
        var seen = Set<String>()
        return groups.flatMap(\.all).filter { seen.insert($0.uid).inserted }
    }

    /// The viewer on every copy of every group loaded, at `item`; paging past the last one loads more groups.
    private func open(_ item: API.ItemCard) {
        let pager = self.pager
        router.show(Self.flat(pager.items), start: item) {
            await pager.loadMore()
            return Self.flat(pager.items).map(\.viewerEntry)
        }
    }

    private func makeBest(_ item: API.ItemCard) async {
        guard working == nil else { return }
        working = item.uid
        defer { working = nil }
        do {
            let api = try Cloud.api()
            try await ReviewDecisions.run("Best copy changed.", message: { (_: API.Ok) in "Best copy changed." }) {
                try await api.chooseBest(uid: item.uid)
            }
            CloudEvents.post(items: [item.uid])
            await pager.refresh()
        } catch {
            Toaster.shared.error(error, doing: "change the best copy")
        }
    }

    private func delete(_ item: API.ItemCard) async {
        deleting = nil
        guard working == nil else { return }
        working = item.uid
        deletingNow = true
        defer {
            working = nil
            deletingNow = false
        }
        do {
            let api = try Cloud.api()
            try await ReviewDecisions.run("Copy moved to the trash.", message: { (_: API.Ok) in "Copy moved to the trash." }) {
                try await api.deleteItem(uid: item.uid)
            }
            CloudEvents.post(items: [item.uid])
            await pager.refresh()
        } catch {
            Toaster.shared.error(error, doing: "delete the copy")
        }
    }
}
