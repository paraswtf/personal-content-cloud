import SwiftUI

/// Every share link (the website's Settings > Sharing, docs/ARCHITECTURE.md section 26): its status, views and
/// expiry, with Copy, Send, Edit (title, password, expiry, downloads) and Revoke. Settings opens it.
struct SharingLinksScreen: View {
    @State private var shares: [API.ShareOut] = []
    @State private var loaded = false
    @State private var loadError: Error?
    @State private var editing: SharingRow?
    @State private var revoking: API.ShareOut?

    /// Live links first, then the expired and revoked ones (as the website lists them).
    private var ordered: [API.ShareOut] {
        let now = Date()
        let live = shares.filter { SharingText.status($0, now: now) == .active }
        let rest = shares.filter { SharingText.status($0, now: now) != .active }
        return live + rest
    }

    var body: some View {
        content
            .navigationTitle("Share links")
            .task { if !loaded { await load() } }
            .refreshable { await load() }
            .onCloudChange { Task { await load() } }
            .sheet(item: $editing) { row in
                SharingEditSheet(share: row.share) { updated in replace(updated) }
            }
            .confirmAction("Take this link down?",
                           isPresented: Binding(get: { revoking != nil }, set: { if !$0 { revoking = nil } }),
                           actionTitle: "Revoke",
                           message: "Anyone who has it will see that it no longer exists. The photos themselves are not touched.") {
                [revoking] in
                if let share = revoking { Task { await revoke(share) } }
            }
    }

    @ViewBuilder
    private var content: some View {
        if !loaded {
            LoadingState()
        } else if shares.isEmpty, let loadError {
            ErrorState(error: loadError) { Task { await load() } }
        } else {
            List {
                Section {
                    if shares.isEmpty {
                        Text("No links yet. Select photos in the library and choose Share, or share an album from its page.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(ordered, id: \.id) { share in
                            row(share)
                        }
                    }
                } footer: {
                    Text("Links you created to show photos to people without an account. Private items are never part of a link.")
                }
            }
            .accessibilityIdentifier("sharing.list")
        }
    }

    private func row(_ share: API.ShareOut) -> some View {
        let status = SharingText.status(share)
        let title = share.title.isEmpty ? "Untitled link" : share.title
        return HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: "link")
                        .foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                    Text(title)
                        .font(.body.weight(.semibold))
                        .lineLimit(1)
                }
                HStack(spacing: 6) {
                    SharingStatusBadge(status: status)
                    if share.hasPassword {
                        Chip(text: "Password", systemImage: "lock.fill", tint: .blue)
                    }
                }
                Text(SharingText.details(share))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .accessibilityElement(children: .combine)
            Spacer(minLength: 0)
            if status != .revoked {
                Menu {
                    actions(share)
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.title3)
                        .frame(width: 44, height: 44)
                }
                .accessibilityLabel("Actions for \(title)")
                .accessibilityIdentifier("sharing.menu")
            }
        }
        .swipeActions(edge: .trailing) {
            if status != .revoked {
                Button(role: .destructive) {
                    revoking = share
                } label: {
                    Label("Revoke", systemImage: "xmark.circle")
                }
                Button {
                    editing = SharingRow(share: share)
                } label: {
                    Label("Edit", systemImage: "pencil")
                }
                .tint(.blue)
            }
        }
        .contextMenu {
            if status != .revoked { actions(share) }
        }
    }

    @ViewBuilder
    private func actions(_ share: API.ShareOut) -> some View {
        Button {
            SharingText.copy(share)
        } label: {
            Label("Copy link", systemImage: "doc.on.doc")
        }
        if let link = SharingText.url(share) {
            ShareLink(item: link) {
                Label("Send link", systemImage: "square.and.arrow.up")
            }
        }
        Button {
            editing = SharingRow(share: share)
        } label: {
            Label("Edit", systemImage: "pencil")
        }
        Button(role: .destructive) {
            revoking = share
        } label: {
            Label("Revoke", systemImage: "xmark.circle")
        }
    }

    private func load() async {
        do {
            shares = try await Cloud.api().listShares()
            loadError = nil
        } catch is CancellationError {
            return
        } catch {
            loadError = error
        }
        loaded = true
    }

    private func replace(_ share: API.ShareOut) {
        if let i = shares.firstIndex(where: { $0.id == share.id }) { shares[i] = share }
    }

    private func revoke(_ share: API.ShareOut) async {
        do {
            try await AlbumsActions.run("The link was taken down.") {
                try await Cloud.api().revokeShare(shareId: share.id)
            }
            await load()
        } catch {
            Toaster.shared.error(SharingText.error(error, fallback: "Could not revoke the link."))
        }
    }
}

/// Edit a share link (the website's EditShareModal): the title, the password (keep, set or change, remove), the
/// expiry (keep, or a new one from now) and the download switch.
struct SharingEditSheet: View {
    enum PasswordChoice: Hashable { case keep, set, clear }
    enum ExpiryChoice: Hashable {
        case keep
        case new(SharingExpiry)
    }

    let share: API.ShareOut
    var onSaved: (API.ShareOut) -> Void = { _ in }

    @Environment(\.dismiss) private var dismiss
    @State private var title = ""
    @State private var passwordChoice: PasswordChoice = .keep
    @State private var password = ""
    @State private var expiryChoice: ExpiryChoice = .keep
    @State private var allowDownload = true
    @State private var prefilled = false
    @State private var busy = false
    @State private var errorText: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("Title") {
                    TextField("Shared photos", text: $title)
                        .accessibilityLabel("Title")
                        .accessibilityIdentifier("sharing.edit.title")
                }
                Section("Password") {
                    Picker("Password", selection: $passwordChoice) {
                        Text(share.hasPassword ? "Keep the current password" : "No password").tag(PasswordChoice.keep)
                        Text(share.hasPassword ? "Change the password" : "Set a password").tag(PasswordChoice.set)
                        if share.hasPassword {
                            Text("Remove the password").tag(PasswordChoice.clear)
                        }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                    if passwordChoice == .set {
                        SecureField("New password", text: $password)
                            .accessibilityLabel("New password")
                            .accessibilityIdentifier("sharing.edit.password")
                    }
                }
                Section {
                    Picker("Expires", selection: $expiryChoice) {
                        Text("Keep (\(SharingText.expiry(share.expiresAt).lowercased()))").tag(ExpiryChoice.keep)
                        ForEach(SharingExpiry.allCases) { choice in
                            Text(choice.days == nil ? choice.label : "\(choice.label) from now").tag(ExpiryChoice.new(choice))
                        }
                    }
                    .accessibilityIdentifier("sharing.edit.expiry")
                    Toggle("Allow downloading the original files", isOn: $allowDownload)
                        .accessibilityIdentifier("sharing.edit.download")
                }
                if let errorText {
                    Section {
                        Text(errorText)
                            .foregroundStyle(.red)
                            .accessibilityIdentifier("sharing.edit.error")
                    }
                }
            }
            .navigationTitle("Edit link")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                        .disabled(busy)
                }
                ToolbarItem(placement: .confirmationAction) {
                    if busy {
                        ProgressView()
                    } else {
                        Button("Save") { Task { await save() } }
                            .accessibilityIdentifier("sharing.edit.save")
                    }
                }
            }
            .onAppear {
                guard !prefilled else { return }
                prefilled = true
                title = share.title
                allowDownload = share.allowDownload
            }
            .onChange(of: title) { _, value in
                if value.count > 120 { title = String(value.prefix(120)) }
            }
        }
        .interactiveDismissDisabled(busy)
    }

    private func save() async {
        guard !busy else { return }
        var update = API.ShareUpdate(title: title.trimmingCharacters(in: .whitespacesAndNewlines),
                                     allowDownload: allowDownload)
        switch passwordChoice {
        case .keep:
            break
        case .set:
            let secret = password.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !secret.isEmpty else {
                errorText = "Enter the new password, or choose to keep or clear the current one."
                return
            }
            update.password = secret
        case .clear:
            update.password = ""
        }
        if case .new(let choice) = expiryChoice {
            if let days = choice.days {
                update.expiresDays = days
            } else {
                update.nullKeys.insert(.expiresDays)
            }
        }
        busy = true
        errorText = nil
        defer { busy = false }
        let request = update
        do {
            let saved = try await AlbumsActions.run("Link updated.") {
                try await Cloud.api().updateShare(shareId: share.id, body: request)
            }
            onSaved(saved)
            dismiss()
        } catch {
            errorText = SharingText.error(error, fallback: "Could not save the changes.")
        }
    }
}
