import SwiftUI
import SwiftData

struct SyncView: View {
    @ObservedObject private var engine = SyncEngine.shared
    @ObservedObject private var uploader = BackgroundUploader.shared
    @Query(sort: \SyncEvent.at, order: .reverse) private var events: [SyncEvent]
    @Query(filter: #Predicate<SyncRecord> { $0.status == "synced" }) private var synced: [SyncRecord]
    @Query(filter: #Predicate<SyncRecord> { $0.status == "failed" }) private var failed: [SyncRecord]
    @Query(filter: #Predicate<SyncRecord> { $0.status == "uploaded" }) private var waiting: [SyncRecord]

    var body: some View {
        NavigationStack {
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
                        Text("\(synced.count) synced, \(failed.count) failed").font(.footnote).foregroundStyle(.secondary)
                    }
                    if !waiting.isEmpty {
                        Text("\(waiting.count) sent, waiting for the server to store them").font(.footnote).foregroundStyle(.secondary)
                    }
                }
                if !failed.isEmpty {
                    Section("Failed") {
                        ForEach(failed.prefix(20)) { r in
                            VStack(alignment: .leading) {
                                Text(r.filename.isEmpty ? r.assetId : r.filename)
                                Text(r.error ?? "").font(.footnote).foregroundStyle(.red)
                            }
                        }
                    }
                }
                Section("Log") {
                    ForEach(events.prefix(100)) { e in
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
