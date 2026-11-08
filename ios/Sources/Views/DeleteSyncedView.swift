import SwiftUI
import Photos

/// Assets the server confirms it holds, with a way to remove them from the phone's library. Only
/// assets confirmed as stored and not in the server's trash are offered by default; those in the
/// trash sit under their own heading so the owner decides (section 28).
struct DeleteSyncedView: View {
    @State private var rows: [SyncEngine.SyncedRow] = []
    @State private var trashedRows: [SyncEngine.SyncedRow] = []
    @State private var selected: Set<String> = []
    @State private var loading = false
    @State private var confirm = false
    @State private var message: String?

    private var allRows: [SyncEngine.SyncedRow] { rows + trashedRows }

    var body: some View {
        NavigationStack {
            Group {
                if loading {
                    ProgressView("Checking with the server")
                } else if allRows.isEmpty {
                    ContentUnavailableView("Nothing to delete", systemImage: "checkmark.circle",
                                           description: Text("Synced photos that are still on this phone appear here once the server confirms it has them."))
                } else {
                    List {
                        if !rows.isEmpty {
                            Section("Stored in your cloud") {
                                ForEach(rows) { row in rowView(row) }
                            }
                        }
                        if !trashedRows.isEmpty {
                            Section {
                                ForEach(trashedRows) { row in rowView(row) }
                            } header: {
                                Text("In your cloud's trash")
                            } footer: {
                                Text("These were deleted on the server and sit in its trash. Removing them from this phone leaves only the trash copy, which goes for good when the trash is emptied.")
                            }
                        }
                    }
                }
            }
            .navigationTitle("Delete synced")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    // "All" selects the stored ones only; trashed ones are picked one by one.
                    Button(!rows.isEmpty && Set(rows.map(\.id)).isSubset(of: selected) ? "None" : "All") {
                        if !rows.isEmpty && Set(rows.map(\.id)).isSubset(of: selected) { selected.removeAll() } else { selected = Set(rows.map(\.id)) }
                    }
                    .disabled(rows.isEmpty)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Delete \(selected.count)", role: .destructive) { confirm = true }
                        .disabled(selected.isEmpty)
                }
            }
            .confirmationDialog(confirmTitle, isPresented: $confirm, titleVisibility: .visible) {
                Button("Remove from this phone", role: .destructive) { Task { await deleteSelected() } }
            }
            .task { await load() }
            .refreshable { await load() }
            .overlay(alignment: .bottom) {
                if let message {
                    Text(message).font(.footnote).padding(10).background(.thinMaterial, in: Capsule()).padding()
                }
            }
        }
    }

    private var confirmTitle: String {
        let inTrash = trashedRows.filter { selected.contains($0.id) }.count
        if inTrash > 0 {
            return "Remove \(selected.count) items from this phone? \(inTrash) of them are only in your cloud's trash."
        }
        return "Remove \(selected.count) items from this phone? They stay in your cloud."
    }

    @ViewBuilder
    private func rowView(_ row: SyncEngine.SyncedRow) -> some View {
        let rec = row.rec
        HStack(spacing: 12) {
            AssetThumb(asset: row.asset).frame(width: 56, height: 56).clipShape(RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading) {
                Text(rec.filename).lineLimit(1)
                Text("\(ByteCountFormatter.string(fromByteCount: rec.bytes, countStyle: .file)) · \(rec.createdAt.formatted(date: .abbreviated, time: .omitted))")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: selected.contains(rec.assetId) ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(selected.contains(rec.assetId) ? Color.accentColor : .secondary)
        }
        .contentShape(Rectangle())
        .onTapGesture { toggle(rec.assetId) }
    }

    private func toggle(_ id: String) {
        if selected.contains(id) { selected.remove(id) } else { selected.insert(id) }
    }

    private func load() async {
        loading = true
        let confirmed = await SyncEngine.shared.confirmedSynced()
        rows = confirmed.stored
        trashedRows = confirmed.trashed
        selected = selected.intersection(allRows.map(\.id))
        loading = false
    }

    private func deleteSelected() async {
        let chosen = allRows.filter { selected.contains($0.id) }
        let ok = await PhotoLibrary.delete(chosen.map(\.asset))
        if ok {
            SyncEngine.shared.markDeleted(chosen.map(\.rec))
            message = "\(chosen.count) removed from this phone"
            SyncStore.shared.log("Removed \(chosen.count) synced items from the library")
            selected.removeAll()
            await load()
        } else {
            message = "Nothing was removed"
        }
    }
}

struct AssetThumb: View {
    let asset: PHAsset
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Color.secondary.opacity(0.15)
            if let image { Image(uiImage: image).resizable().scaledToFill() }
        }
        .task(id: asset.localIdentifier) {
            let options = PHImageRequestOptions()
            options.deliveryMode = .fastFormat
            options.isNetworkAccessAllowed = false
            PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: 112, height: 112),
                                                  contentMode: .aspectFill, options: options) { img, _ in
                if let img { image = img }
            }
        }
    }
}
