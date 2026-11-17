import SwiftUI

/// Makes a share link (the website's ShareModal, docs/ARCHITECTURE.md section 26) for chosen items or an album:
/// title, password, expiry and the download switch, then the link to copy or send. Presented as a sheet by the
/// album screen, the library's selection and the viewer:
/// `.sheet(isPresented: $sharing) { SharingCreateSheet(target: .items(uids: uids)) }`.
struct SharingCreateSheet: View {
    let target: SharingTarget

    @Environment(\.dismiss) private var dismiss
    @State private var title = ""
    @State private var password = ""
    @State private var expiry: SharingExpiry = .never
    @State private var allowDownload = true
    @State private var created: API.ShareOut?
    @State private var prefilled = false
    @State private var busy = false
    @State private var errorText: String?

    var body: some View {
        NavigationStack {
            Group {
                if let created {
                    result(created)
                } else {
                    form
                }
            }
            .navigationTitle(created == nil ? "Share" : "Share link")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    if created == nil {
                        Button("Cancel") { dismiss() }
                            .disabled(busy)
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    if created != nil {
                        Button("Done") { dismiss() }
                            .accessibilityIdentifier("sharing.done")
                    } else if busy {
                        ProgressView()
                    } else {
                        Button("Create link") { Task { await create() } }
                            .accessibilityIdentifier("sharing.create")
                    }
                }
            }
            .onAppear {
                guard !prefilled else { return }
                prefilled = true
                if case .album(_, let albumTitle) = target { title = albumTitle }
            }
        }
        .interactiveDismissDisabled(busy)
    }

    private var form: some View {
        Form {
            Section {
                Text("Make a link that shows \(SharingText.what(target)) to someone without an account. Private items are left out.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Section("Title") {
                TextField("Shared photos", text: $title)
                    .accessibilityLabel("Title")
                    .accessibilityIdentifier("sharing.title")
            }
            Section {
                SecureField("Password", text: $password)
                    .accessibilityLabel("Password")
                    .accessibilityIdentifier("sharing.password")
            } header: {
                Text("Password (optional)")
            }
            Section {
                Picker("Expires", selection: $expiry) {
                    ForEach(SharingExpiry.allCases) { choice in
                        Text(choice.label).tag(choice)
                    }
                }
                .accessibilityIdentifier("sharing.expiry")
                Toggle("Allow downloading the original files", isOn: $allowDownload)
                    .accessibilityIdentifier("sharing.download")
            }
            if let errorText {
                Section {
                    Text(errorText)
                        .foregroundStyle(.red)
                        .accessibilityIdentifier("sharing.error")
                }
            }
        }
        .onChange(of: title) { _, value in
            if value.count > 120 { title = String(value.prefix(120)) }
        }
        .onChange(of: password) { _, value in
            if value.count > 200 { password = String(value.prefix(200)) }
        }
    }

    private func result(_ share: API.ShareOut) -> some View {
        let link = SharingText.url(share)
        return Form {
            Section {
                Text("Anyone with this link can see \(SharingText.what(target))\(share.hasPassword ? " after entering the password" : ""). Your private items are never part of a link.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Section("Link") {
                Text(link?.absoluteString ?? share.path)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
                    .accessibilityLabel("Share link")
                    .accessibilityValue(link?.absoluteString ?? share.path)
                    .accessibilityIdentifier("sharing.link")
                Button {
                    SharingText.copy(share)
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
                .accessibilityIdentifier("sharing.copy")
                if let link {
                    ShareLink(item: link) {
                        Label("Send", systemImage: "square.and.arrow.up")
                    }
                    .accessibilityIdentifier("sharing.send")
                }
            }
            Section {
                if case .album = target, share.count == 0 {
                    Text("This album has no photos the link can show yet.")
                        .foregroundStyle(.orange)
                }
                Text("\(Fmt.count(share.count, "item")). You can change the password, the expiry or take the link down under Settings, Share links.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func request() -> API.ShareCreate {
        var share = API.ShareCreate(kind: .items)
        switch target {
        case .album(let id, _):
            share.kind = .album
            share.albumId = id
        case .items(let uids):
            share.kind = .items
            share.itemUids = uids
        }
        share.title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let secret = password.trimmingCharacters(in: .whitespacesAndNewlines)
        share.password = secret.isEmpty ? nil : secret
        share.expiresDays = expiry.days
        share.allowDownload = allowDownload
        return share
    }

    private func create() async {
        guard !busy else { return }
        busy = true
        errorText = nil
        defer { busy = false }
        let made = request()
        do {
            let share = try await AlbumsActions.run("Link created.") {
                try await Cloud.api().createShare(body: made)
            }
            created = share
        } catch {
            errorText = SharingText.error(error, fallback: "The link could not be created.")
        }
    }
}
