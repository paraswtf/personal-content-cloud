import SwiftUI

/// Settings > Undo history (the website's UndoHistory, docs/ARCHITECTURE.md sections 14 and 33): every change this
/// phone made, newest first (UndoCenter keeps the list), with Undo or Redo on each, and Clear history (this phone
/// only; the server keeps what each action changed for 30 days either way).
struct SettingsUndoHistoryView: View {
    @State private var busy: String?
    @State private var confirmClear = false

    var body: some View {
        let entries = UndoCenter.shared.entries
        Group {
            if entries.isEmpty {
                EmptyState(title: "Nothing to undo yet", systemImage: "arrow.uturn.backward.circle",
                           text: "The changes you make on this phone show up here, newest first, and can be undone for 30 days.")
            } else {
                TimelineView(.periodic(from: .now, by: 30)) { context in
                    List {
                        Section {
                            ForEach(entries) { entry in
                                row(entry, now: context.date)
                            }
                        } header: {
                            Text("Changes made on this phone, newest first.")
                                .textCase(nil)
                        }
                    }
                    .accessibilityIdentifier("undo.list")
                }
            }
        }
        .navigationTitle("Undo history")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if !entries.isEmpty {
                    Button("Clear history") { confirmClear = true }
                        .accessibilityIdentifier("undo.clear")
                }
            }
        }
        .confirmAction("Forget all \(entries.count) on this phone?", isPresented: $confirmClear, actionTitle: "Clear",
                       message: "The changes stay on the server; only this list is cleared.") {
            UndoCenter.shared.clear()
        }
        #if DEBUG
        .task { await SettingsSystemTestHook.makeUndoableChangeIfAsked() }
        #endif
    }

    private func row(_ e: UndoCenter.Entry, now: Date) -> some View {
        let canUndo = e.state == .done && !e.expired
        let canRedo = e.state == .undone && e.undoId != nil && !e.expired
        return HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(e.label)
                    .font(.subheadline.weight(.medium))
                    .strikethrough(e.state == .undone)
                    .foregroundStyle(e.state == .undone ? Color.secondary : Color.primary)
                Text(SettingsSystemText.undoStatus(e, now: now))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .combine)
            Spacer(minLength: 8)
            if canUndo || canRedo {
                Button {
                    Task { await run(e, redo: canRedo) }
                } label: {
                    if busy == e.id {
                        ProgressView()
                    } else {
                        Label(canUndo ? "Undo" : "Redo", systemImage: canUndo ? "arrow.uturn.backward" : "arrow.uturn.forward")
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(busy != nil && busy != e.id)
                .accessibilityLabel("\(canUndo ? "Undo" : "Redo"): \(e.label)")
            }
        }
    }

    private func run(_ e: UndoCenter.Entry, redo: Bool) async {
        guard busy == nil else { return }
        busy = e.id
        if redo {
            await UndoCenter.shared.redo(e.id)
        } else {
            await UndoCenter.shared.undo(e.id)
        }
        busy = nil
    }
}
