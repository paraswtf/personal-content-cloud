import SwiftUI
import UIKit

/// The queue's totals and Clear finished (the website's SummaryBar), the first row of the queue.
struct UploadSummaryRow: View {
    let summary: UploadSummary
    let progress: Double
    let onClearFinished: () -> Void

    var body: some View {
        let busy = summary.sending > 0 || summary.processing > 0
        HStack(alignment: .top, spacing: 10) {
            if busy {
                ProgressView().padding(.top, 2)
            } else {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).font(.title3)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(summary.headline(progress: progress))
                    .font(.subheadline.weight(.semibold))
                    .accessibilityIdentifier("upload.headline")
                if !summary.outcomes.isEmpty {
                    Text(summary.outcomes).font(.subheadline).foregroundStyle(.secondary)
                        .accessibilityIdentifier("upload.outcomes")
                }
                if summary.sending > 0 {
                    Text("Uploads keep going in the background once they have started. Analysis and indexing continue on the server after that.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
            if summary.finished > 0 {
                Button("Clear finished", action: onClearFinished)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .accessibilityIdentifier("upload.clearFinished")
            }
        }
        .padding(.vertical, 2)
    }
}

/// One file of the queue (the website's QueueRow): its picture, name, size, phase, progress, outcome and buttons.
struct UploadQueueRow: View {
    let rec: SyncRecord
    let info: UploadRowInfo
    let resumed: Bool
    let onView: (() -> Void)?
    let onRetry: () -> Void
    let onCancel: () -> Void
    let onRemove: () -> Void

    private var name: String { rec.filename.isEmpty ? "Untitled" : rec.filename }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            UploadThumb(rec: rec)
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .top, spacing: 4) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(name).font(.subheadline.weight(.medium)).lineLimit(1).truncationMode(.middle)
                        HStack(spacing: 4) {
                            if info.phase == .checking || info.phase == .processing {
                                ProgressView().controlSize(.mini)
                            }
                            Text(subtitle).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Spacer(minLength: 4)
                    buttons
                }
                if info.phase == .uploading {
                    ProgressView(value: min(1, max(0, info.progress ?? 0)))
                        .accessibilityLabel("\(name): \(UploadText.phase(info.phase, progress: info.progress, resumed: resumed) ?? "")")
                }
                if info.phase == .waiting, let problem = rec.error {
                    Label(problem, systemImage: "wifi.exclamationmark").font(.caption).foregroundStyle(.orange)
                }
                if let badge = UploadText.badge(phase: info.phase, serverStatus: rec.serverStatus, skipped: info.skipped) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        UploadBadge(text: badge, tone: tone)
                        if info.phase != .cancelled, let message = rec.error, !message.isEmpty {
                            Text(message).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .padding(.vertical, 2)
    }

    private var subtitle: String {
        var parts: [String] = []
        if rec.origin == "camera" { parts.append("Camera") }
        if rec.bytes > 0 { parts.append(Fmt.bytes(rec.bytes)) }
        if let phase = UploadText.phase(info.phase, progress: info.progress, resumed: resumed) { parts.append(phase) }
        return parts.joined(separator: " · ")
    }

    private var tone: UploadBadge.Tone {
        switch info.phase {
        case .done:
            switch rec.serverStatus {
            case "stored": return .ok
            case "duplicate": return .neutral
            default: return .warn
            }
        case .failed: return .warn
        default: return .neutral
        }
    }

    @ViewBuilder private var buttons: some View {
        HStack(spacing: 0) {
            if let onView {
                Button("View", action: onView)
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 8)
                    .frame(minHeight: 44)
                    .accessibilityLabel("View \(name)")
                    .accessibilityIdentifier("upload.view")
            }
            if info.canRetry {
                iconButton("arrow.counterclockwise", label: "Try \(name) again", id: "upload.retry", action: onRetry)
            }
            if info.phase.canCancel {
                iconButton("xmark", label: "Cancel \(name)", id: "upload.cancel", action: onCancel)
            }
            if info.phase.isFinished {
                iconButton("trash", label: "Remove \(name) from the list", id: "upload.remove", action: onRemove)
            }
        }
        .buttonStyle(.borderless)
    }

    private func iconButton(_ systemImage: String, label: String, id: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage).frame(width: 40, height: 44)
        }
        .foregroundStyle(.secondary)
        .accessibilityLabel(label)
        .accessibilityIdentifier(id)
    }
}

/// An outcome as a small capsule (the website's StatusBadge).
struct UploadBadge: View {
    enum Tone { case neutral, ok, warn }

    let text: String
    let tone: Tone

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon).imageScale(.small).accessibilityHidden(true)
            Text(text).accessibilityIdentifier("upload.status")
        }
        .font(.caption.weight(.medium))
        .foregroundStyle(color)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(color.opacity(0.12), in: Capsule())
    }

    private var color: Color {
        switch tone {
        case .neutral: return .secondary
        case .ok: return .green
        case .warn: return .orange
        }
    }

    private var icon: String {
        switch text {
        case "Added": return "checkmark.circle"
        case "Already in your library": return "doc.on.doc"
        case "Not accepted": return "nosign"
        case "Cancelled": return "xmark.circle"
        default: return "exclamationmark.circle"
        }
    }
}

/// The picture of a queued file: the server's thumb once it has one, else a small picture made on the phone, else
/// the kind's icon (with the extension for documents).
struct UploadThumb: View {
    let rec: SyncRecord

    @State private var local: UIImage?

    var body: some View {
        let kind = UploadKind.of(name: rec.filename)
        ZStack {
            RoundedRectangle(cornerRadius: 10).fill(Color(.secondarySystemBackground))
            if let ref = rec.thumbRef {
                RemoteImage(ref: ref, maxPixel: 160, vault: rec.itemVault)
            } else if let local {
                Image(uiImage: local).resizable().scaledToFill()
            } else {
                VStack(spacing: 2) {
                    Image(systemName: kind.systemImage).font(.title3)
                    if kind == .pdf || kind == .document {
                        Text(String(UploadKind.fileExtension(rec.filename).prefix(4)).uppercased())
                            .font(.system(size: 9, weight: .bold))
                    }
                }
                .foregroundStyle(.secondary)
            }
        }
        .frame(width: 56, height: 56)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .accessibilityHidden(true)
        .task(id: "\(rec.assetId)|\(rec.status)|\(rec.serverStatus ?? "")") {
            if local == nil { local = UploadThumbs.image(rec.assetId) }
        }
    }
}
