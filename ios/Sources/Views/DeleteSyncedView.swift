import SwiftUI
import Photos

/// Assets the server confirms it holds, with a way to remove them from the phone's library.
struct DeleteSyncedView: View {
    @State private var rows: [SyncEngine.SyncedRow] = []
    @State private var selected: Set<String> = []
    @State private var loading = false
    @State private var confirm = false
    @State private var message: String?

    var body: some View {
        NavigationStack {
            Group {
                if loading {
                    ProgressView("Checking with the server")
                } else if rows.isEmpty {
                    ContentUnavailableView("Nothing to delete", systemImage: "checkmark.circle",
                                           description: Text("Synced photos that are still on this phone appear here once the server confirms it has them."))
                } else {
                    List(rows) { row in
                        let asset = row.asset
                        let rec = row.rec
                        HStack(spacing: 12) {
                            AssetThumb(asset: asset).frame(width: 56, height: 56).clipShape(RoundedRectangle(cornerRadius: 8))
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
                }
            }
            .navigationTitle("Delete synced")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button(selected.count == rows.count && !rows.isEmpty ? "None" : "All") {
                        if selected.count == rows.count { selected.removeAll() } else { selected = Set(rows.map(\.id)) }
                    }
                    .disabled(rows.isEmpty)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Delete \(selected.count)", role: .destructive) { confirm = true }
                        .disabled(selected.isEmpty)
                }
            }
            .confirmationDialog("Remove \(selected.count) items from this phone? They stay in your cloud.",
                                isPresented: $confirm, titleVisibility: .visible) {
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

    private func toggle(_ id: String) {
        if selected.contains(id) { selected.remove(id) } else { selected.insert(id) }
    }

    private func load() async {
        loading = true
        rows = await SyncEngine.shared.confirmedSynced()
        selected = selected.intersection(rows.map(\.id))
        loading = false
    }

    private func deleteSelected() async {
        let chosen = rows.filter { selected.contains($0.id) }
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
