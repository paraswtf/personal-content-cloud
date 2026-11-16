import SwiftUI

/// Settings > System (the website's SystemTab with StorageCards, VideoCard and ThroughputCard): the processor, the
/// server, the models, the processing queue, video copies, storage, backups, processing speed, failed uploads and
/// housekeeping, from GET /api/system asked again every 8 seconds while the screen is open. Missing models can be
/// downloaded from here; housekeeping runs the cleanup jobs now; failed uploads can be queued again. These are
/// server jobs, not changes to the library, so they have no undo (docs/ARCHITECTURE.md section 33).
struct SettingsSystemView: View {
    @State private var system: API.SystemOut?
    @State private var loadError: Error?
    @State private var refreshing = false
    @State private var downloading = false
    @State private var cleaning = false
    @State private var retrying = false
    @State private var backingUp = false
    @State private var detecting = false
    @State private var requeueing = false

    var body: some View {
        Group {
            if let system {
                content(system, SettingsSystemStatus(system))
            } else if let loadError {
                ErrorState(error: loadError) { Task { await load() } }
            } else {
                LoadingState(text: "Loading system status")
            }
        }
        .navigationTitle("System")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            while !Task.isCancelled {
                await load()
                try? await Task.sleep(nanoseconds: 8_000_000_000)
            }
        }
    }

    private func content(_ s: API.SystemOut, _ x: SettingsSystemStatus) -> some View {
        List {
            processorSection(x)
            serverSection(s)
            modelsSection(s, x)
            queueSection(x)
            if let video = x.video { videoSection(video) }
            storageSection(s, x)
            backupSection(s, x)
            SettingsThroughputSection()
            failedUploadsSection
            housekeepingSection
        }
        .accessibilityIdentifier("system.list")
        .refreshable { await load() }
    }

    // MARK: Sections

    private func processorSection(_ x: SettingsSystemStatus) -> some View {
        Section {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "cpu")
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(x.gpu.name ?? (x.gpu.available ? "GPU" : "No GPU found"))
                        .font(.body.weight(.semibold))
                    Text(x.gpu.available
                         ? "Models run on \(SettingsSystemText.providerLabel(x.gpu.provider))."
                         : "Models run on the CPU. Analysis works, just more slowly.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            .accessibilityElement(children: .combine)
        } header: {
            SettingsSectionHeader(title: "Processor") {
                SettingsBadge(text: x.gpu.available ? "GPU in use" : "CPU only", tone: x.gpu.available ? .ok : .warn)
            }
        }
    }

    private func serverSection(_ s: API.SystemOut) -> some View {
        Section {
            LabeledContent("Version", value: s.version)
            LabeledContent("Time zone", value: s.timezone)
            LabeledContent("Background workers") {
                SettingsDotText(text: s.workers ? "Running" : "Stopped", tone: s.workers ? .ok : .danger)
            }
            LabeledContent("ffmpeg (videos)") {
                SettingsDotText(text: s.ffmpeg ? "Found" : "Missing", tone: s.ffmpeg ? .ok : .danger)
            }
            if !s.ffmpeg {
                Text("Without ffmpeg, videos are stored but get no previews, frames or faces.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            SettingsActionRow(title: "Refresh", systemImage: "arrow.clockwise", busy: refreshing) {
                Task {
                    refreshing = true
                    await load()
                    refreshing = false
                }
            }
            .accessibilityIdentifier("system.refresh")
        } header: {
            SettingsSectionHeader(title: "Server") { EmptyView() }
        }
    }

    private func modelsSection(_ s: API.SystemOut, _ x: SettingsSystemStatus) -> some View {
        let missing = SettingsSystemStatus.missingModels(s)
        let running = x.downloadRunning(s)
        return Section {
            if missing > 0 || running {
                SettingsActionRow(title: SettingsSystemText.downloadLabel(missing: missing, running: running),
                                  systemImage: "arrow.down.circle", busy: downloading || running) {
                    Task { await download() }
                }
                .accessibilityIdentifier("system.models.download")
            }
            if let dl = x.download, dl.running || dl.progress != nil || dl.error != nil {
                Text(dl.error.map { "Download failed: \($0)" } ?? dl.progress ?? "Downloading")
                    .font(.footnote)
                    .foregroundStyle(dl.error != nil ? Color.red : Color.secondary)
            }
            ForEach(s.models, id: \.key) { model in
                SettingsModelRow(model: model)
            }
        } header: {
            SettingsSectionHeader(title: "Models") {
                if missing == 0 && !running { SettingsBadge(text: "All present", tone: .ok) }
            }
        } footer: {
            Text("All analysis runs on this server. Nothing is sent to outside services.")
        }
    }

    private func queueSection(_ x: SettingsSystemStatus) -> some View {
        Section {
            HStack(alignment: .top) {
                SettingsStat(label: "Waiting", value: x.queues.queued)
                SettingsStat(label: "Running", value: x.queues.running)
                SettingsStat(label: "Failed", value: x.queues.failed, danger: x.queues.failed > 0)
            }
            ForEach(x.queues.byQueue) { line in
                LabeledContent {
                    Text("\(line.queued.formatted()) waiting, \(line.running.formatted()) running")
                        .monospacedDigit()
                } label: {
                    Text(SettingsSystemText.queueLabel(line.name))
                }
            }
            if x.queues.failed > 0 {
                Text("Failed jobs were retried and gave up. Open the item and choose to process it again, or run housekeeping.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        } header: {
            SettingsSectionHeader(title: "Processing queue") {
                SettingsBadge(text: x.queues.busy ? "Working" : "Idle", tone: x.queues.busy ? .warn : .ok)
            }
        }
    }

    private func videoSection(_ v: SettingsSystemStatus.Video) -> some View {
        Section {
            VStack(alignment: .leading, spacing: 4) {
                Text(SettingsSystemText.encoderLine(v))
                    .font(.subheadline.weight(.semibold))
                if let reason = v.reason {
                    Text("Not used: \(reason)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            HStack(alignment: .top) {
                SettingsStat(label: "Done", value: v.done)
                SettingsStat(label: "Pending", value: v.pending)
                SettingsStat(label: "Queued", value: v.queued)
                SettingsStat(label: "Running", value: v.running)
                SettingsStat(label: "Failed", value: v.failed, danger: v.failed > 0)
            }
            SettingsActionRow(title: "Test encoders", systemImage: "waveform.badge.magnifyingglass", busy: detecting) {
                Task { await detect() }
            }
            .disabled(requeueing)
            .accessibilityIdentifier("system.video.detect")
            SettingsActionRow(title: "Re-run failed copies", systemImage: "arrow.counterclockwise", busy: requeueing) {
                Task { await requeue() }
            }
            .disabled(detecting || v.failed == 0)
            .accessibilityIdentifier("system.video.requeue")
        } header: {
            SettingsSectionHeader(title: "Video") {
                if v.encoder != nil {
                    SettingsBadge(text: v.hw ? "Hardware encoder" : "Software encoder", tone: v.hw ? .ok : .warn)
                }
            }
        } footer: {
            Text("Videos a browser or a phone cannot play as they are get an H.264 web copy, up to 1080p, made with the machine's encoder.")
        }
    }

    private func storageSection(_ s: API.SystemOut, _ x: SettingsSystemStatus) -> some View {
        Section {
            ForEach(SettingsTier.tiers(s)) { tier in
                SettingsTierRow(tier: tier)
            }
            LabeledContent("Original files", value: Fmt.bytes(s.disk.originalsBytes))
            if let m = x.migration, m.active || m.running {
                VStack(alignment: .leading, spacing: 6) {
                    Text(SettingsSystemText.migrationLine(m))
                        .font(.subheadline)
                    if m.running {
                        ProgressView(value: Double(m.percent), total: 100)
                            .accessibilityLabel("Originals moved")
                    }
                    if let error = m.error {
                        Text("Last problem: \(error)")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }
            }
        } header: {
            SettingsSectionHeader(title: "Storage") { EmptyView() }
        } footer: {
            Text(s.storage?.tiered == true
                 ? "Originals live on their own disk; the database, previews and thumbnails stay with the app."
                 : "One folder holds the database, the originals and the previews. Set ORIGINALS_DIR to put the originals on another disk.")
        }
    }

    @ViewBuilder
    private func backupSection(_ s: API.SystemOut, _ x: SettingsSystemStatus) -> some View {
        if let b = s.backup, b.enabled {
            Section {
                Text(SettingsSystemText.backupLastLine(x.backupLast))
                    .font(.subheadline)
                    .foregroundStyle(x.backupLast.map { !$0.ok } == true ? Color.red : Color.secondary)
                if b.running, let p = x.backupProgress {
                    VStack(alignment: .leading, spacing: 6) {
                        ProgressView(value: Double(p.percent), total: 100)
                            .accessibilityLabel("Backup progress")
                        Text(SettingsSystemText.backupPhase(p))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                LabeledContent("Next run", value: b.nextRunAt.map { "\(Fmt.date($0, withTime: true)) (\(Fmt.until($0)))" }
                    ?? "every night at \(SettingsSystemText.hour(b.hour))")
                LabeledContent("Snapshots kept", value: "\(b.snapshots.count.formatted()) of \(b.keep)")
                VStack(alignment: .leading, spacing: 2) {
                    Text("Backup folder")
                    Text(b.dir ?? "")
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    if let total = b.totalBytes, let free = b.freeBytes {
                        Text("\(Fmt.bytes(free)) free of \(Fmt.bytes(total)) on that disk")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if !b.running {
                    SettingsActionRow(title: "Back up now", systemImage: "archivebox", busy: backingUp) {
                        Task { await backUp() }
                    }
                    .accessibilityIdentifier("system.backup.now")
                }
            } header: {
                SettingsSectionHeader(title: "Backups") {
                    if b.running { SettingsBadge(text: "Running", tone: .warn) }
                }
            } footer: {
                Text("Every night at \(SettingsSystemText.hour(b.hour)): the database, the originals, the share-link key, the learned models and the Wallet certificates. Previews and thumbnails are not copied; they are rebuilt from the originals.")
            }
        } else {
            Section {
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: "archivebox")
                        .foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                    Text("Set BACKUP_DIR to a folder on another disk and restart. A snapshot of the database, the originals and the learned models is then taken every night; the newest ones are kept and unchanged files are shared between them.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            } header: {
                SettingsSectionHeader(title: "Backups") { SettingsBadge(text: "Off", tone: .warn) }
            }
        }
    }

    private var failedUploadsSection: some View {
        Section {
            SettingsActionRow(title: "Retry failed uploads", systemImage: "arrow.counterclockwise", busy: retrying) {
                Task { await retryFailed() }
            }
            .accessibilityIdentifier("system.retryFailed")
        } header: {
            SettingsSectionHeader(title: "Failed uploads") { EmptyView() }
        } footer: {
            Text("Uploads whose processing failed, for example because ffmpeg was missing or timed out, are kept. Retry them once the cause is fixed.")
        }
    }

    private var housekeepingSection: some View {
        Section {
            SettingsActionRow(title: "Run housekeeping", systemImage: "server.rack", busy: cleaning) {
                Task { await housekeeping() }
            }
            .accessibilityIdentifier("system.housekeeping")
        } header: {
            SettingsSectionHeader(title: "Housekeeping") { EmptyView() }
        } footer: {
            Text("Runs the cleanup tasks now: unfinished uploads that expired, trash past its keep period and files no item uses. It also runs on its own schedule.")
        }
    }

    // MARK: Server calls

    private func load() async {
        do {
            let answer = try await Cloud.api().system()
            system = answer
            loadError = nil
        } catch {
            if SettingsSystemView.cancelled(error) { return }
            if system == nil { loadError = error }
        }
    }

    nonisolated static func cancelled(_ error: Error) -> Bool {
        error is CancellationError || (error as? URLError)?.code == .cancelled
    }

    private func download() async {
        downloading = true
        defer { downloading = false }
        do {
            _ = try await Cloud.api().downloadModels()
            Toaster.shared.ok("Model download started. Progress shows here.")
            await load()
        } catch {
            Toaster.shared.error(error, doing: "start the download")
        }
    }

    private func housekeeping() async {
        cleaning = true
        defer { cleaning = false }
        do {
            _ = try await Cloud.api().runHousekeeping()
            Toaster.shared.ok("Housekeeping started.")
            CloudEvents.post()
            await load()
        } catch {
            Toaster.shared.error(error, doing: "run housekeeping")
        }
    }

    private func retryFailed() async {
        retrying = true
        defer { retrying = false }
        do {
            let out = try await Cloud.api().retryFailed()
            Toaster.shared.ok(SettingsSystemText.retryMessage(out.requeued))
            await load()
        } catch {
            Toaster.shared.error(error, doing: "retry the failed uploads")
        }
    }

    private func backUp() async {
        backingUp = true
        defer { backingUp = false }
        do {
            let out = try await Cloud.api().backupNow()
            if out.started {
                Toaster.shared.ok("Backup started. Progress shows here.")
            } else if out.running {
                Toaster.shared.ok("A backup is already running.")
            } else {
                Toaster.shared.ok("Backups are off: set BACKUP_DIR first.")
            }
            await load()
        } catch {
            Toaster.shared.error(error, doing: "start the backup")
        }
    }

    private func detect() async {
        detecting = true
        defer { detecting = false }
        do {
            let raw = try await Cloud.api().videoDetect()
            let v = SettingsSystemStatus.Video(raw)
            Toaster.shared.ok(v.encoder != nil ? "Web copies will use \(SettingsSystemText.encoderLine(v))." : "No encoder answered.")
            await load()
        } catch {
            Toaster.shared.error(error, doing: "test the encoders")
        }
    }

    private func requeue() async {
        requeueing = true
        defer { requeueing = false }
        do {
            let out = try await Cloud.api().videoRequeue(body: API.VideoRequeueIn(scope: .failed))
            Toaster.shared.ok(SettingsSystemText.requeueMessage(out.requeued))
            await load()
        } catch {
            Toaster.shared.error(error, doing: "queue the copies")
        }
    }
}

/// One model: its name, its state, its size and where it runs once loaded.
struct SettingsModelRow: View {
    let model: API.ModelStatus

    var body: some View {
        let state = SettingsSystemText.modelState(model)
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "sparkles")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .padding(.top, 2)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(model.label.isEmpty ? model.key : model.label)
                    .font(.subheadline.weight(.semibold))
                if let error = model.error, !error.isEmpty {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                HStack(spacing: 10) {
                    SettingsDotText(text: state.text, tone: state.tone)
                    if model.sizeBytes > 0 { Text(Fmt.bytes(model.sizeBytes)) }
                    if model.loaded { Text(SettingsSystemText.providerLabel(model.provider)) }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

/// One disk: what it holds, its free space and how full it is.
struct SettingsTierRow: View {
    let tier: SettingsTier

    var body: some View {
        let title = SettingsSystemText.tierTitle(tier.key)
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Label(title, systemImage: "internaldrive")
                    .font(.subheadline.weight(.semibold))
                Spacer(minLength: 8)
                if let shared = tier.sharedWith {
                    Text("Same disk as \(SettingsSystemText.tierTitle(shared).lowercased())")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("\(Fmt.bytes(tier.freeBytes)) free of \(Fmt.bytes(tier.totalBytes))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(tier.lowSpace ? Color.red : Color.secondary)
                }
            }
            if !tier.path.isEmpty {
                Text(tier.path)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .truncationMode(.middle)
            }
            Text("Holds \(tier.holds).")
                .font(.caption)
                .foregroundStyle(.secondary)
            if tier.sharedWith == nil {
                ProgressView(value: Double(tier.usedPercent), total: 100)
                    .tint(tier.lowSpace ? Color.red : Color.accentColor)
                    .accessibilityLabel("\(title) disk used")
                Text(SettingsSystemText.tierUsage(tier))
                    .font(.caption)
                    .foregroundStyle(tier.lowSpace ? Color.red : Color.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}
