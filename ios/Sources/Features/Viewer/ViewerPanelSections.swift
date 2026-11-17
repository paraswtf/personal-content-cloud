import SwiftUI
import UIKit

/// Tags (the website's components/viewer/TagsPanel.tsx, docs/ARCHITECTURE.md section 19): the owner's tags, the
/// learner's suggestions (accept makes them the owner's, remove teaches the learner), and a field that offers
/// existing tags as you type and creates a new one when nothing matches. Person tags are in the people section.
struct ViewerTagsSection: View {
    let context: ViewerPageContext
    let item: API.ItemDetail

    @Environment(AppRouter.self) private var router
    @State private var text = ""
    @State private var options: [API.TagOut] = []
    @State private var busy = false

    private var typed: String { text.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var mine: [API.ItemTagOut] { item.userTags.filter { $0.source == .user } }
    private var suggested: [API.ItemTagOut] { item.userTags.filter { $0.source == .auto } }

    var body: some View {
        Section("Tags") {
            if mine.isEmpty && suggested.isEmpty {
                Text("No tags yet.").foregroundStyle(.secondary)
            }
            if !mine.isEmpty {
                ViewerFlow {
                    ForEach(mine, id: \.id) { tag in
                        HStack(spacing: 2) {
                            if linksToLibrary {
                                Button {
                                    showTagged(tag.name)
                                } label: {
                                    Text(tag.name).lineLimit(1).foregroundStyle(.primary)
                                }
                                .accessibilityLabel(tag.name)
                                .accessibilityHint("Shows everything tagged \(tag.name)")
                                .accessibilityIdentifier("viewer.tag.show")
                            } else {
                                Text(tag.name).lineLimit(1)
                            }
                            Button {
                                Task { await remove(tag) }
                            } label: {
                                Image(systemName: "xmark").imageScale(.small).frame(width: 28, height: 28)
                            }
                            .accessibilityLabel("Remove tag \(tag.name)")
                        }
                        .font(.subheadline.weight(.medium))
                        .padding(.leading, 12)
                        .padding(.trailing, 4)
                        .background(Color.secondary.opacity(0.16), in: Capsule())
                        .accessibilityElement(children: .contain)
                        .accessibilityIdentifier("viewer.tag")
                    }
                }
                .buttonStyle(.borderless)
                .disabled(busy)
            }
            if !suggested.isEmpty {
                Text("Suggested from items you tagged that look alike:")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                ViewerFlow {
                    ForEach(suggested, id: \.id) { tag in
                        HStack(spacing: 2) {
                            Text(tag.name).lineLimit(1)
                            if let score = tag.score { Text(Fmt.percent(score)).foregroundStyle(.secondary) }
                            Button {
                                Task { await add(API.UserTagIn(tagId: tag.id), name: tag.name) }
                            } label: {
                                Image(systemName: "checkmark").imageScale(.small).frame(width: 28, height: 28)
                            }
                            .accessibilityLabel("Accept suggested tag \(tag.name)")
                            Button {
                                Task { await remove(tag) }
                            } label: {
                                Image(systemName: "xmark").imageScale(.small).frame(width: 28, height: 28)
                            }
                            .accessibilityLabel("Remove suggested tag \(tag.name)")
                        }
                        .font(.subheadline)
                        .padding(.leading, 12)
                        .padding(.trailing, 4)
                        .overlay(Capsule().strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4, 3])).foregroundStyle(.secondary))
                    }
                }
                .buttonStyle(.borderless)
                .disabled(busy)
            }
            HStack {
                TextField("Add a tag, like wedding", text: $text)
                    .textInputAutocapitalization(.never)
                    .submitLabel(.done)
                    .onSubmit { Task { await addTyped() } }
                    .accessibilityIdentifier("viewer.tags.field")
                Button("Add") { Task { await addTyped() } }
                    .buttonStyle(.borderless)
                    .disabled(typed.isEmpty || busy)
                    .accessibilityIdentifier("viewer.tags.add")
            }
            if !options.isEmpty {
                ViewerFlow {
                    ForEach(options, id: \.id) { tag in
                        Button {
                            Task { await add(API.UserTagIn(tagId: tag.id), name: tag.name) }
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "plus").imageScale(.small)
                                Text(tag.name).lineLimit(1)
                                Text("\(tag.items)")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                                .font(.subheadline)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.secondary.opacity(0.12), in: Capsule())
                        }
                        .accessibilityLabel("Add tag \(tag.name), \(Fmt.count(tag.items, "item"))")
                    }
                }
                .buttonStyle(.borderless)
            }
            if !typed.isEmpty, !options.contains(where: { same($0.name, typed) }), !mine.contains(where: { same($0.name, typed) }) {
                Text("New tag. Press Add to create \(typed).")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .task(id: typed) { await loadOptions() }
    }

    /// A tag opens the library with that tag (the website's link). The vault's items stay in the vault, whose
    /// screen has no tag filter on the phone, so their tags are plain text.
    private var linksToLibrary: Bool { !context.vault && !item.nsfw }

    /// Everything with the tag: the viewer closes and the library opens filtered on the tab underneath, as Show on
    /// map does.
    private func showTagged(_ name: String) {
        var filters = LibraryFilters()
        filters.tag = [name]
        router.viewer = nil
        router.open(.libraryFiltered(filters))
    }

    private func same(_ a: String, _ b: String) -> Bool {
        a.trimmingCharacters(in: .whitespaces).lowercased() == b.trimmingCharacters(in: .whitespaces).lowercased()
    }

    private func loadOptions() async {
        let q = typed
        guard !q.isEmpty else { options = []; return }
        try? await Task.sleep(nanoseconds: 200_000_000)
        guard !Task.isCancelled, q == typed else { return }
        let onItem = Set(mine.map(\.id))
        let list = (try? await Cloud.api().listTags(q: q)) ?? []
        guard q == typed else { return }
        options = Array(list.filter { !onItem.contains($0.id) }.prefix(8))
    }

    private func addTyped() async {
        let name = typed
        guard !name.isEmpty, !busy else { return }
        if mine.contains(where: { same($0.name, name) }) { text = ""; return }
        let fromList = options.first(where: { same($0.name, name) }).map { (id: $0.id, name: $0.name) }
        let fromSuggested = suggested.first(where: { same($0.name, name) }).map { (id: $0.id, name: $0.name) }
        if let known = fromList ?? fromSuggested {
            await add(API.UserTagIn(tagId: known.id), name: known.name)
        } else {
            await add(API.UserTagIn(name: String(name.prefix(40))), name: name)
        }
    }

    private func add(_ body: API.UserTagIn, name: String) async {
        busy = true
        defer { busy = false }
        let done = await ViewerActions.run(context, "Tag \(name) added.", doing: "add the tag") {
            _ = try await $0.addItemTag(uid: context.uid, body: body)
        }
        if done { text = ""; options = [] }
    }

    private func remove(_ tag: API.ItemTagOut) async {
        busy = true
        defer { busy = false }
        let label = tag.source == .auto ? "Suggestion \(tag.name) removed." : "Tag \(tag.name) removed."
        await ViewerActions.run(context, label, doing: "remove the tag") {
            _ = try await $0.removeItemTag(uid: context.uid, tagId: tag.id)
        }
    }
}

/// "Maybe related to" (the website's components/viewer/RelatedSuggest.tsx, section 10): people this item may be
/// related to, found from the items the owner marked as related to them. Add marks it related; No is remembered for
/// that person. Only named people are offered.
struct ViewerRelatedSection: View {
    let context: ViewerPageContext

    @State private var suggestions: [API.ItemRelatedSuggestion] = []
    @State private var busy = false

    var body: some View {
        Group {
            if !suggestions.isEmpty {
                Section("Maybe related to") {
                    ForEach(suggestions, id: \.person.id) { suggestion in
                        let name = suggestion.person.name ?? ""
                        HStack(spacing: 8) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(name).font(.body.weight(.semibold))
                                Text(why(suggestion.reason)).font(.footnote).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button {
                                Task { await addRelated(suggestion.person.id) }
                            } label: {
                                Image(systemName: "checkmark").frame(width: 36, height: 36)
                            }
                            .accessibilityLabel("Add as related to \(name)")
                            Button {
                                Task { await dismiss(suggestion.person.id) }
                            } label: {
                                Image(systemName: "xmark").frame(width: 36, height: 36)
                            }
                            .accessibilityLabel("Not related to \(name)")
                        }
                        .buttonStyle(.borderless)
                        .disabled(busy)
                    }
                }
            }
        }
        .task(id: context.json) { await load() }
    }

    private func why(_ reason: API.Reason) -> String {
        switch reason {
        case .alike: return "looks like one of their related items"
        case .moment: return "same moment as one of their related items"
        case .place: return "same place as one of their related items"
        default: return "like one of their related items"
        }
    }

    private func load() async {
        let answer = try? await APIContext.$vault.withValue(context.vault) {
            try await Cloud.api().itemRelatedSuggestions(uid: context.uid)
        }
        suggestions = (answer?.people ?? []).filter { Fmt.isNamed($0.person.name) }
    }

    private func addRelated(_ personId: Int) async {
        busy = true
        defer { busy = false }
        let done = await ViewerActions.run(context, "Tagged as related", doing: "tag") {
            _ = try await $0.addTag(uid: context.uid, body: API.TagIn(personId: personId, relation: .related))
        }
        if done { await load() }
    }

    private func dismiss(_ personId: Int) async {
        busy = true
        defer { busy = false }
        let done = await ViewerActions.run(context, "Not suggested for them again", doing: "save that") {
            _ = try await $0.dismissRelated(personId: personId, body: API.RelatedDismissIn(uids: [context.uid]))
        }
        if done { await load() }
    }
}

/// The text read from the item (the website's TextPanel): its length and where it came from, folded to a few lines
/// with Show all, and a copy button.
struct ViewerTextSection: View {
    let context: ViewerPageContext

    @State private var text: API.ItemTextOut?
    @State private var failed = false
    @State private var expanded = false
    @State private var copied = false

    var body: some View {
        Group {
            if let text {
                let content = text.text.trimmingCharacters(in: .whitespacesAndNewlines)
                if content.isEmpty {
                    Text("No text was found.").foregroundStyle(.secondary)
                } else {
                    HStack {
                        Text("\(Fmt.count(text.charCount, "character")), \(ViewerFormat.textSource(text.source))")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button {
                            UIPasteboard.general.string = content
                            copied = true
                            Task {
                                try? await Task.sleep(nanoseconds: 1_500_000_000)
                                copied = false
                            }
                        } label: {
                            Image(systemName: copied ? "checkmark" : "doc.on.doc").frame(width: 36, height: 36)
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel("Copy the text")
                        .accessibilityIdentifier("viewer.text.copy")
                    }
                    let long = ViewerFormat.isLongText(content)
                    Text(content.replacingOccurrences(of: "\u{0C}", with: "\n\n"))
                        .font(.callout)
                        .lineLimit(expanded || !long ? nil : 8)
                        .textSelection(.enabled)
                        .accessibilityIdentifier("viewer.text.body")
                    if long {
                        Button(expanded ? "Show less" : "Show all") { expanded.toggle() }
                            .buttonStyle(.borderless)
                    }
                }
            } else if failed {
                Text("The text could not be loaded.").foregroundStyle(.secondary)
            } else {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading text").foregroundStyle(.secondary)
                }
            }
        }
        .task(id: context.uid) { await load() }
    }

    private func load() async {
        do {
            text = try await APIContext.$vault.withValue(context.vault) {
                try await Cloud.api().itemText(uid: context.uid)
            }
            failed = false
        } catch {
            if !(error is CancellationError) { failed = true }
        }
    }
}

/// The albums the item is in (the website's components/albums/AlbumsSection.tsx, section 24): each opens its album,
/// can make the item its cover or take it out, and Add to album opens the albums area's sheet.
struct ViewerAlbumsSection: View {
    let context: ViewerPageContext
    let item: API.ItemDetail

    @Environment(AppRouter.self) private var router
    @State private var adding = false
    @State private var busy = false

    var body: some View {
        Section("Albums") {
            if item.albums.isEmpty {
                Text("Not in an album yet.").foregroundStyle(.secondary)
            }
            ForEach(item.albums, id: \.id) { album in
                HStack(spacing: 8) {
                    Button {
                        router.viewer = nil
                        router.open(.album(id: album.id))
                    } label: {
                        Label(album.name, systemImage: "rectangle.stack").lineLimit(1)
                    }
                    .accessibilityLabel("Open the album \(album.name)")
                    Spacer()
                    Button {
                        Task { await setCover(album) }
                    } label: {
                        Image(systemName: "star").frame(width: 36, height: 36)
                    }
                    .accessibilityLabel("Use as the cover of \(album.name)")
                    Button {
                        Task { await removeFrom(album) }
                    } label: {
                        Image(systemName: "xmark").frame(width: 36, height: 36)
                    }
                    .accessibilityLabel("Remove from \(album.name)")
                }
                .buttonStyle(.borderless)
                .disabled(busy)
            }
            Button {
                adding = true
            } label: {
                Label("Add to album", systemImage: "plus")
            }
            .disabled(busy)
            .accessibilityIdentifier("viewer.info.addToAlbum")
        }
        .sheet(isPresented: $adding) {
            AlbumsAddSheet(uids: [context.uid]) {
                context.reload()
                CloudEvents.post(items: [context.uid])
            }
        }
    }

    private func setCover(_ album: API.AlbumRef) async {
        busy = true
        defer { busy = false }
        await ViewerActions.run(context, "Now the cover of \(album.name).", doing: "set the cover") {
            _ = try await $0.updateAlbum(albumId: album.id, body: API.AlbumUpdate(coverItemUid: context.uid))
        }
    }

    private func removeFrom(_ album: API.AlbumRef) async {
        busy = true
        defer { busy = false }
        await ViewerActions.run(context, "Removed from \(album.name).", doing: "change the album") {
            _ = try await $0.bulk(body: API.BulkIn(uids: [context.uid], action: .unalbum, value: String(album.id)))
        }
    }
}
