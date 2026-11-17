import SwiftUI

/// What the selection bar asks the browser to open.
enum LibraryBulkSheet: String, Identifiable {
    case album, share, people, tag, untag

    var id: String { rawValue }
}

/// The bar shown while selecting (the website's components/library/BulkBar.tsx): Done, All, how many are picked,
/// and every action: favourite, keep, forwards, category, people, tags, private, albums, share and delete. The
/// library, an album and the vault share it; the album swaps Add to album for Remove from album, and the vault has no
/// category or share.
struct LibraryBulkBar: View {
    let model: LibraryModel
    let open: (LibraryBulkSheet) -> Void
    let delete: () -> Void

    var body: some View {
        let selection = model.selection
        let count = selection.count
        let scope = model.scope
        let allFavorite = model.allSelectedFavorite
        let anyForward = model.anySelectedForward
        VStack(spacing: 8) {
            HStack(spacing: 16) {
                Button("Done") { selection.end() }
                    .fontWeight(.semibold)
                    .accessibilityIdentifier("bulk.done")
                Button("All") { selection.select(model.loadedUids) }
                    .accessibilityLabel("Select all")
                    .accessibilityIdentifier("library.bulk.all")
                Spacer()
                if model.bulkBusy {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("Working")
                }
                Text(count == 0 ? "Select items" : Fmt.count(count, "selected", "selected"))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("bulk.count")
            }
            .padding(.horizontal, 16)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    action(allFavorite ? "Unfavourite" : "Favourite", allFavorite ? "star.slash" : "star",
                           "library.bulk.favorite") {
                        run(allFavorite ? .unfavorite : .favorite)
                    }
                    if model.anySelectedMisclick {
                        action("Keep", "checkmark.circle", "library.bulk.keep") { run(.keep) }
                    }
                    action(anyForward ? "Not forwards" : "Forwards", "arrowshape.turn.up.right",
                           "library.bulk.forwards") {
                        run(anyForward ? .unforward : .forward)
                    }
                    if !scope.isVault {
                        Menu {
                            ForEach(LibraryFilters.categories) { choice in
                                Button(choice.label) { run(.category, value: choice.value.rawValue) }
                            }
                        } label: {
                            Label("Set category", systemImage: "folder")
                        }
                        .menuStyle(.button)
                        .buttonStyle(.bordered)
                        .disabled(count == 0 || model.bulkBusy)
                        .accessibilityIdentifier("library.bulk.category")
                    }
                    action("People", "person.2", "library.bulk.people") { open(.people) }
                    action("Add tag", "tag", "library.bulk.tag") { open(.tag) }
                    action("Remove tag", "tag.slash", "library.bulk.untag") { open(.untag) }
                    action(scope.isVault ? "Not private" : "Mark private", scope.isVault ? "lock.open" : "lock",
                           "library.bulk.private") {
                        run(.nsfw, value: scope.isVault ? "0" : "1")
                    }
                    if let albumId = scope.albumId {
                        action("Remove from album", "rectangle.stack.badge.minus", "library.bulk.album") {
                            run(.unalbum, value: String(albumId))
                        }
                    } else {
                        action("Add to album", "rectangle.stack.badge.plus", "library.bulk.album") { open(.album) }
                    }
                    if !scope.isVault {
                        action("Share", "square.and.arrow.up", "library.bulk.share") { open(.share) }
                    }
                    action("Delete", "trash", "library.bulk.delete", role: .destructive) { delete() }
                }
                .padding(.horizontal, 16)
            }
        }
        .padding(.vertical, 10)
        .background(.bar)
    }

    private func run(_ action: API.Action, value: String? = nil) {
        Task { await model.bulk(action, value: value) }
    }

    private func action(_ title: String, _ systemImage: String, _ identifier: String, role: ButtonRole? = nil,
                        perform: @escaping () -> Void) -> some View {
        Button(role: role, action: perform) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
        }
        .buttonStyle(.bordered)
        .disabled(model.selection.count == 0 || model.bulkBusy)
        .accessibilityIdentifier(identifier)
    }
}

/// Add tag or Remove tag for the selection (the website's TagForm): a name, new or existing, with the existing
/// tags to pick from.
struct LibraryTagSheet: View {
    let removing: Bool
    let onSubmit: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var text = ""
    @State private var tags: [API.TagOut] = []

    private var name: String { text.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    TextField(removing ? "Tag name" : "Tag name, new or existing", text: $text)
                        .submitLabel(.done)
                        .onSubmit(submit)
                        .accessibilityLabel(removing ? "Tag to remove" : "Tag to add")
                        .accessibilityIdentifier("library.tag.name")
                }
                let matches = tags.filter { name.isEmpty || $0.name.localizedCaseInsensitiveContains(name) }
                if !matches.isEmpty {
                    Section("Tags") {
                        ForEach(matches, id: \.id) { tag in
                            Button(tag.name) { text = tag.name }
                                .foregroundStyle(.primary)
                        }
                    }
                }
            }
            .navigationTitle(removing ? "Remove a tag" : "Add a tag")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button(removing ? "Remove" : "Add", action: submit)
                        .disabled(name.isEmpty)
                        .accessibilityIdentifier("library.tag.submit")
                }
            }
            .onChange(of: text) { _, value in
                if value.count > LibraryFilters.maxTagLength { text = String(value.prefix(LibraryFilters.maxTagLength)) }
            }
            .task { tags = (try? await Cloud.api().listTags()) ?? [] }
        }
        .presentationDetents([.medium, .large])
    }

    private func submit() {
        guard !name.isEmpty else { return }
        dismiss()
        onSubmit(name)
    }
}
