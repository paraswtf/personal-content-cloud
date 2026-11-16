import SwiftUI

/// New album and Rename (the website's AlbumDialog): a name and a description.
struct AlbumsEditSheet: View {
    enum Mode {
        case create
        case edit(API.AlbumOut)
    }

    let mode: Mode
    /// Runs with the saved album, before the sheet closes.
    var onSaved: (API.AlbumOut) -> Void = { _ in }

    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var description = ""
    @State private var prefilled = false
    @State private var busy = false
    @State private var errorText: String?
    @FocusState private var nameFocused: Bool

    private static let nameLimit = 120
    private static let descriptionLimit = 2000

    private var isCreate: Bool {
        if case .create = mode { return true }
        return false
    }

    private var trimmedName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        NavigationStack {
            Form {
                Section("Name") {
                    TextField("Goa trip", text: $name)
                        .focused($nameFocused)
                        .submitLabel(.done)
                        .onSubmit { Task { await save() } }
                        .accessibilityLabel("Name")
                        .accessibilityIdentifier("albums.edit.name")
                }
                Section("Description") {
                    TextField("Optional", text: $description, axis: .vertical)
                        .lineLimit(3...8)
                        .accessibilityLabel("Description")
                        .accessibilityIdentifier("albums.edit.description")
                }
                if let errorText {
                    Section {
                        Text(errorText)
                            .foregroundStyle(.red)
                            .accessibilityIdentifier("albums.edit.error")
                    }
                }
            }
            .navigationTitle(isCreate ? "New album" : "Rename album")
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
                        Button(isCreate ? "Create" : "Save") { Task { await save() } }
                            .disabled(trimmedName.isEmpty)
                            .accessibilityIdentifier("albums.edit.save")
                    }
                }
            }
            .onChange(of: name) { _, value in
                if value.count > Self.nameLimit { name = String(value.prefix(Self.nameLimit)) }
            }
            .onChange(of: description) { _, value in
                if value.count > Self.descriptionLimit { description = String(value.prefix(Self.descriptionLimit)) }
            }
            .onAppear {
                guard !prefilled else { return }
                prefilled = true
                if case .edit(let album) = mode {
                    name = album.name
                    description = album.description
                }
                nameFocused = true
            }
        }
        .interactiveDismissDisabled(busy)
    }

    private func save() async {
        let newName = trimmedName
        guard !newName.isEmpty, !busy else { return }
        let newDescription = description.trimmingCharacters(in: .whitespacesAndNewlines)
        busy = true
        errorText = nil
        defer { busy = false }
        do {
            let saved: API.AlbumOut
            switch mode {
            case .create:
                saved = try await AlbumsActions.run("Album \(newName) created.") {
                    try await Cloud.api().createAlbum(body: API.AlbumIn(name: newName, description: newDescription))
                }
            case .edit(let album):
                saved = try await AlbumsActions.run("Album updated.") {
                    try await Cloud.api().updateAlbum(albumId: album.id,
                                                      body: API.AlbumUpdate(name: newName, description: newDescription))
                }
            }
            CloudEvents.post()
            onSaved(saved)
            dismiss()
        } catch {
            errorText = AlbumsText.error(error, doing: isCreate ? "create the album" : "update the album")
        }
    }
}
