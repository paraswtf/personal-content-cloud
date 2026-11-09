import SwiftUI
import SwiftData

struct SyncView: View {
    @ObservedObject private var engine = SyncEngine.shared
    @ObservedObject private var uploader = BackgroundUploader.shared
    @Environment(\.modelContext) private var ctx
    // Only what the screen shows is loaded. Loading every record (or every log line) made each save
    // during a large sync reload thousands of objects on the main thread, and a log line trimmed
    // by SyncStore.log while still listed here crashed the app.
    @Query(SyncView.recentEvents) private var events: [SyncEvent]
    @Query(SyncView.recentFailed) private var failed: [SyncRecord]

    static var recentEvents: FetchDescriptor<SyncEvent> {
        var d = FetchDescriptor<SyncEvent>(sortBy: [SortDescriptor(\.at, order: .reverse)])
        d.fetchLimit = 100
        return d
    }

    static var recentFailed: FetchDescriptor<SyncRecord> {
        var d = FetchDescriptor<SyncRecord>(predicate: #Predicate<SyncRecord> { $0.status == "failed" },
                                            sortBy: [SortDescriptor(\.updatedAt, order: .reverse)])
        d.fetchLimit = 20
        return d
    }

    private func count(_ status: String) -> Int {
        (try? ctx.fetchCount(FetchDescriptor<SyncRecord>(predicate: #Predicate<SyncRecord> { $0.status == status }))) ?? 0
    }

    var body: some View {
        let syncedCount = count("synced"), failedCount = count("failed"), waitingCount = count("uploaded")
        return NavigationStack {
            List {
                Section {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(engine.statusLine).font(.headline)
                            if let at = AppSettings.lastSyncAt {
                                Text("Last run \(at.formatted(.relative(presentation: .named)))")
                                    .font(.footnote).foregroundStyle(.secondary)
                            }
                        }
                        Spacer()
                        if engine.running {
                            ProgressView(value: engine.progress).frame(width: 80)
                        }
                    }
                    if uploader.sending.count > 1 {
                        Text("Sending \(uploader.sending.count) files at once").font(.footnote).foregroundStyle(.secondary)
                    }
                    ForEach(uploader.sendingList) { file in
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Sending \(file.filename)").font(.footnote).lineLimit(1)
                            ProgressView(value: file.progress)
                        }
                    }
                    HStack {
                        Button(engine.running || !uploader.sending.isEmpty ? "Stop" : "Sync now") {
                            if engine.running || !uploader.sending.isEmpty { engine.stop() } else { engine.start(manual: true) }
                        }
                        .buttonStyle(.borderedProminent)
                        Spacer()
                        Text("\(syncedCount) synced, \(failedCount) failed").font(.footnote).foregroundStyle(.secondary)
                    }
                    if waitingCount > 0 {
                        Text("\(waitingCount) sent, waiting for the server to store them").font(.footnote).foregroundStyle(.secondary)
                    }
                }
                if !failed.isEmpty {
                    Section("Failed") {
                        ForEach(failed) { r in
                            VStack(alignment: .leading) {
                                Text(r.filename.isEmpty ? r.assetId : r.filename)
                                Text(r.error ?? "").font(.footnote).foregroundStyle(.red)
                            }
                        }
                    }
                }
                Section("Log") {
                    ForEach(events) { e in
                        HStack(alignment: .top) {
                            Text(e.at, style: .time).font(.footnote).foregroundStyle(.secondary).frame(width: 64, alignment: .leading)
                            Text(e.text).font(.footnote).foregroundStyle(e.level == "error" ? .red : .primary)
                        }
                    }
                }
            }
            .navigationTitle("Sync")
            .onAppear { if AppSettings.autoSync { engine.start(manual: false) } }
        }
    }
}
