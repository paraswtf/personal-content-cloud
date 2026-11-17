import SwiftUI

/// What a selection of a person's items opens.
enum PeopleItemsSheet: Identifiable {
    case people([String])
    case album([String])
    case share([String])

    var id: String {
        switch self {
        case .people: return "people"
        case .album: return "album"
        case .share: return "share"
        }
    }
}

/// A person's Items or Related tab: their items in the media grid (the viewer pages through them), and a selection
/// for People (move, remove, tag or rename who the items show), albums and sharing. The Related tab first suggests
/// items that may be related too.
struct PeoplePersonItemsTab: View {
    let person: API.PersonOut
    let related: Bool
    let header: AnyView
    /// A rename in the People sheet merged this person into someone else: the page shows that person.
    var onMerged: (Int) -> Void = { _ in }

    @Environment(AppRouter.self) private var router
    @State private var pager: Pager<API.ItemCard>
    @State private var selection = SelectionModel<String>()
    @State private var sheet: PeopleItemsSheet?
    /// Related tab: items that may be related too (loaded here: a view that shows nothing until it has them
    /// would never start loading them).
    @State private var suggested: [API.RelatedSuggestionOut] = []

    init(person: API.PersonOut, related: Bool, header: AnyView, onMerged: @escaping (Int) -> Void = { _ in }) {
        self.person = person
        self.related = related
        self.header = header
        self.onMerged = onMerged
        let id = person.id
        _pager = State(initialValue: Pager<API.ItemCard> { cursor in
            try await Cloud.api().personItems(personId: id, relation: related ? .related : nil, cursor: cursor,
                                              limit: 60).pagerPage
        })
    }

    var body: some View {
        MediaGrid(items: pager.items, selection: selection,
                  onOpen: { open($0) },
                  onAppear: { pager.loadMoreIfNeeded($0) }) {
            VStack(alignment: .leading, spacing: 12) {
                header
                if related && !suggested.isEmpty { PeopleRelatedSuggestions(person: person, items: $suggested) }
                if !pager.items.isEmpty && !selection.active {
                    HStack(spacing: 10) {
                        Button {
                            selection.begin()
                        } label: {
                            Label("Select", systemImage: "checklist")
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("person.items.select")
                        Spacer()
                    }
                    .padding(.horizontal, 16)
                }
                if selection.active {
                    Text("Pick items, then People to move, remove, tag or rename who they show.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 16)
                }
                if pager.items.isEmpty { states.padding(.top, 24) }
            }
            .padding(.bottom, 8)
        }
        .safeAreaInset(edge: .bottom) {
            if selection.active {
                BulkBar(count: selection.count, actions: actions,
                        onSelectAll: { selection.select(pager.items.map(\.uid)) },
                        onDone: { selection.end() })
            }
        }
        .sheet(item: $sheet) { s in
            switch s {
            case .people(let uids):
                PeopleSelectionSheet(uids: uids, fromPersonId: person.id, onDone: { selection.end() },
                                     onMerged: onMerged)
            case .album(let uids):
                AlbumsAddSheet(uids: uids, onDone: { selection.end() })
            case .share(let uids):
                SharingCreateSheet(target: .items(uids: uids))
            }
        }
        .task {
            if !pager.loaded { await pager.refresh() }
            await loadSuggested()
        }
        .refreshable {
            await pager.refresh()
            await loadSuggested()
        }
        .onCloudChange {
            Task {
                await pager.refresh()
                await loadSuggested()
            }
        }
    }

    private func loadSuggested() async {
        guard related else { return }
        if let items = try? await Cloud.api().relatedSuggestions(personId: person.id).items { suggested = items }
    }

    private var chosen: [String] { pager.items.map(\.uid).filter { selection.contains($0) } }

    private var actions: [BulkAction] {
        [
            BulkAction("People in these items", systemImage: "person.2") { sheet = .people(chosen) },
            BulkAction("Add to album", systemImage: "rectangle.stack.badge.plus") { sheet = .album(chosen) },
            BulkAction("Share", systemImage: "square.and.arrow.up") { sheet = .share(chosen) },
        ]
    }

    @ViewBuilder private var states: some View {
        if let error = pager.error {
            ErrorState(error: error) { Task { await pager.retry() } }
        } else if pager.loaded && !pager.loading {
            if related {
                EmptyState(title: "Nothing related yet", systemImage: "link",
                           text: "Open an item, choose \"Tag someone\" and turn on \"Not in the picture, related to them\" to put it here.")
            } else {
                EmptyState(title: "No items yet", systemImage: "photo.on.rectangle")
            }
        } else {
            LoadingState()
        }
    }

    private func open(_ item: API.ItemCard) {
        router.show(pager.items, start: item) {
            await pager.loadMore()
            return pager.items.map(\.viewerEntry)
        }
    }
}

/// Items that may be related to a person or pet too, found from the items the owner marked as related: they look
/// alike, or were taken at the same moment or place. "Add" marks one related, "Not related" is remembered.
struct PeopleRelatedSuggestions: View {
    let person: API.PersonOut
    @Binding var items: [API.RelatedSuggestionOut]

    @Environment(AppRouter.self) private var router
    @State private var busy = false

    private var who: String {
        if Fmt.isNamed(person.name), let name = person.name { return name }
        return PeopleText.isPet(person) ? "this pet" : "this person"
    }

    var body: some View {
        Group {
            if !items.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(alignment: .top) {
                        PeopleSectionHeader(title: "Maybe related to \(who)", systemImage: "sparkles",
                                            note: "Found from the items you marked as related: they look alike, or were taken at the same moment or place.")
                        if items.count > 1 {
                            Button("Add all") { Task { await add(items.map(\.item.uid)) } }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                                .disabled(busy)
                        }
                    }
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(alignment: .top, spacing: 10) {
                            ForEach(items, id: \.item.uid) { s in tile(s) }
                        }
                    }
                }
                .padding(12)
                .background(RoundedRectangle(cornerRadius: 16).fill(Color(uiColor: .secondarySystemBackground)))
                .padding(.horizontal, 16)
            }
        }
    }

    private func reason(_ r: API.Reason) -> String {
        switch r {
        case .alike: return "Looks like one of the related items"
        case .moment: return "From the same moment as one"
        case .place: return "From the same place as one"
        default: return "Related to one of the items"
        }
    }

    private func tile(_ s: API.RelatedSuggestionOut) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                router.show(items.map(\.item), start: s.item)
            } label: {
                Color.clear
                    .aspectRatio(1, contentMode: .fit)
                    .overlay { RemoteImage(ref: s.item.urls.thumb, maxPixel: 384) }
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Open \(s.item.filename)")
            Text(reason(s.reason)).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            HStack(spacing: 6) {
                Button("Add") { Task { await add([s.item.uid]) } }
                    .buttonStyle(.borderedProminent)
                    .accessibilityLabel("Add \(s.item.filename) as related to \(who)")
                Button("Not related") { Task { await no([s.item.uid]) } }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Not related: \(s.item.filename) (\(who))")
            }
            .controlSize(.small)
            .disabled(busy)
        }
        .frame(width: 150)
    }

    private func add(_ uids: [String]) async {
        guard !busy else { return }
        busy = true
        defer { busy = false }
        let who = self.who
        let r = await PeopleAction.run("Marked items as related", doing: "mark them", items: uids) {
            try await Cloud.api().tagPeopleOnItems(body: API.PeopleTagIn(personId: person.id, relation: .related, uids: uids))
        } message: { r in
            r.updated > 0 ? "\(Fmt.count(r.updated, "item")) marked as related to \(who)." : "Already related to \(who)."
        }
        if r != nil { items.removeAll { uids.contains($0.item.uid) } }
    }

    private func no(_ uids: [String]) async {
        guard !busy else { return }
        busy = true
        defer { busy = false }
        let who = self.who
        let r = await PeopleAction.run("Not related", doing: "save that") {
            try await Cloud.api().dismissRelated(personId: person.id, body: API.RelatedDismissIn(uids: uids))
        } message: { _ in
            "Not suggested for \(who) again."
        }
        if r != nil { items.removeAll { uids.contains($0.item.uid) } }
    }
}

/// A person's Faces tab: their faces (tap opens where it was found), and a selection to move them to someone else,
/// take them off the person, or use one as the cover.
struct PeoplePersonFacesTab: View {
    let person: API.PersonOut
    let header: AnyView

    @Environment(AppRouter.self) private var router
    @State private var pager: Pager<PeopleFaceRow>
    @State private var selection = SelectionModel<Int>()
    @State private var moving = false
    @State private var busy = false

    init(person: API.PersonOut, header: AnyView) {
        self.person = person
        self.header = header
        let id = person.id
        _pager = State(initialValue: Pager<PeopleFaceRow> { cursor in
            let page = try await Cloud.api().personFaces(personId: id, cursor: cursor, limit: 120)
            return .init(items: page.items.map { PeopleFaceRow(face: $0) }, next: page.nextCursor, total: page.total)
        })
    }

    private var name: String { PeopleText.name(person) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                HStack(spacing: 10) {
                    if !selection.active && !pager.items.isEmpty {
                        Button {
                            selection.begin()
                        } label: {
                            Label("Select faces", systemImage: "checklist")
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("person.faces.select")
                    }
                    Text(selection.active ? "Pick faces to move, remove, or use as the cover."
                                          : "Tap a face to open where it was found.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 16)
                if pager.items.isEmpty {
                    states.padding(.top, 24)
                } else {
                    PeopleFaceGrid(faces: pager.items, selection: selection, coverId: person.coverFaceId,
                                   onOpen: { open($0) }, onAppear: { pager.loadMoreIfNeeded($0) })
                        .padding(.horizontal, 12)
                }
            }
            .padding(.bottom, 16)
        }
        .safeAreaInset(edge: .bottom) {
            if selection.active {
                BulkBar(count: selection.count, actions: actions,
                        onSelectAll: { selection.select(pager.items.map(\.id)) },
                        onDone: { selection.end() })
            }
        }
        .sheet(isPresented: $moving) {
            PersonPickerSheet(title: "Move \(Fmt.count(selection.count, "face")) to",
                              like: .faces(Array(selection.selected)), excluding: [person.id]) { pick in
                Task { await assign(PeopleChoice(pick)) }
            }
        }
        .task { if !pager.loaded { await pager.refresh() } }
        .refreshable { await pager.refresh() }
        .onCloudChange { Task { await pager.refresh() } }
    }

    private var actions: [BulkAction] {
        [
            BulkAction("Use as cover", systemImage: "star", disabled: selection.count != 1 || busy) {
                Task { await cover() }
            },
            BulkAction("Move to", systemImage: "arrow.right.circle", disabled: busy) { moving = true },
            BulkAction("Not \(name)", systemImage: "person.crop.circle.badge.minus", role: .destructive, disabled: busy) {
                Task { await assign(nil) }
            },
        ]
    }

    @ViewBuilder private var states: some View {
        if let error = pager.error {
            ErrorState(error: error) { Task { await pager.retry() } }
        } else if pager.loaded && !pager.loading {
            EmptyState(title: "No faces", systemImage: "face.dashed",
                       text: "This person was tagged by hand, or by clothes or a tattoo only.")
        } else {
            LoadingState()
        }
    }

    /// The viewer on the items of every face loaded, at this face's item (a video at the face's frame); paging past
    /// the last one loads the next page of faces.
    private func open(_ face: API.FaceListItem) {
        let pager = self.pager
        PeopleViewer.open(router, entries: pager.items.map { PeopleViewer.entry($0.face) }, start: face.itemUid,
                          startAt: face.frameTimeS) {
            await PeopleViewer.more(pager) { PeopleViewer.entry($0.face) }
        }
    }

    private func assign(_ choice: PeopleChoice?) async {
        let ids = Array(selection.selected)
        guard !ids.isEmpty, !busy else { return }
        busy = true
        defer { busy = false }
        let name = self.name
        let r = await PeopleAction.run(choice == nil ? "Took faces off a person" : "Moved faces", doing: "move the faces") {
            try await Cloud.api().assignFaces(body: API.AssignFacesIn(faceIds: ids, personId: choice?.personId,
                                                                      newPersonName: choice?.newName))
        } message: { r in
            let tail = r.followed > 0 ? " \(Fmt.count(r.followed, "look-alike face")) went the same way." : ""
            if choice != nil {
                let to = r.person.map { " to \(PeopleText.name($0))" } ?? ""
                return "\(Fmt.count(r.updated, "face")) moved\(to).\(tail)\(Fmt.guessedTail(r.guessed))"
            }
            return "\(Fmt.count(r.updated, "face")) removed from \(name).\(tail)"
        }
        if r != nil {
            selection.clear()
            await pager.refresh()
        }
    }

    private func cover() async {
        guard let faceId = selection.selected.first, !busy else { return }
        busy = true
        defer { busy = false }
        let r = await PeopleAction.run("Changed the cover", doing: "change the cover") {
            try await Cloud.api().updatePerson(personId: person.id, body: API.PersonUpdate(coverFaceId: faceId))
        } message: { _ in
            "Cover changed."
        }
        if r != nil { selection.clear() }
    }
}

/// A pet's Crops tab: its animal crops, and a selection to move them to another pet, take them off, or remove the
/// boxes (they are not pets).
struct PeoplePersonCropsTab: View {
    let person: API.PersonOut
    let header: AnyView

    @Environment(AppRouter.self) private var router
    @State private var pager: Pager<PeopleCropRow>
    @State private var selection = SelectionModel<Int>()
    @State private var moving = false
    @State private var removing = false
    @State private var busy = false

    init(person: API.PersonOut, header: AnyView) {
        self.person = person
        self.header = header
        let id = person.id
        _pager = State(initialValue: Pager<PeopleCropRow> { cursor in
            let page = try await Cloud.api().personAnimals(personId: id, cursor: cursor, limit: 120)
            return .init(items: page.items.map { PeopleCropRow(crop: $0) }, next: page.nextCursor, total: page.total)
        })
    }

    private var name: String { PeopleText.name(person) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                HStack(spacing: 10) {
                    if !selection.active && !pager.items.isEmpty {
                        Button {
                            selection.begin()
                        } label: {
                            Label("Select crops", systemImage: "checklist")
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("person.crops.select")
                    }
                    Text(selection.active ? "Pick crops to move or remove." : "Tap a crop to open where it was found.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 16)
                if pager.items.isEmpty {
                    states.padding(.top, 24)
                } else {
                    PeopleCropGrid(crops: pager.items, selection: selection, onOpen: { open($0) },
                                   onAppear: { pager.loadMoreIfNeeded($0) })
                        .padding(.horizontal, 12)
                }
            }
            .padding(.bottom, 16)
        }
        .safeAreaInset(edge: .bottom) {
            if selection.active {
                BulkBar(count: selection.count, actions: actions,
                        onSelectAll: { selection.select(pager.items.map(\.id)) },
                        onDone: { selection.end() })
            }
        }
        .sheet(isPresented: $moving) {
            PersonPickerSheet(title: "Move \(Fmt.count(selection.count, "crop")) to", kind: .pet,
                              like: .animals(Array(selection.selected)), excluding: [person.id],
                              species: person.species) { pick in
                Task { await assign(PeopleChoice(pick)) }
            }
        }
        .confirmAction("Remove \(Fmt.count(selection.count, "box", "boxes"))?", isPresented: $removing,
                       actionTitle: "Remove",
                       message: "These are not pets. The boxes are removed from the photos and will not come back.") {
            Task { await removeBoxes() }
        }
        .task { if !pager.loaded { await pager.refresh() } }
        .refreshable { await pager.refresh() }
        .onCloudChange { Task { await pager.refresh() } }
    }

    private var actions: [BulkAction] {
        [
            BulkAction("Move to", systemImage: "arrow.right.circle", disabled: busy) { moving = true },
            BulkAction("Not \(name)", systemImage: "minus.circle", disabled: busy) { Task { await assign(nil) } },
            BulkAction("Remove boxes", systemImage: "trash", role: .destructive, disabled: busy) { removing = true },
        ]
    }

    @ViewBuilder private var states: some View {
        if let error = pager.error {
            ErrorState(error: error) { Task { await pager.retry() } }
        } else if pager.loaded && !pager.loading {
            EmptyState(title: "No crops yet", systemImage: "pawprint")
        } else {
            LoadingState()
        }
    }

    /// The viewer on the items of every crop loaded, at this crop's item (a video at the crop's frame); paging past
    /// the last one loads the next page of crops.
    private func open(_ crop: API.AnimalListItem) {
        let pager = self.pager
        PeopleViewer.open(router, entries: pager.items.map { PeopleViewer.entry($0.crop) }, start: crop.itemUid,
                          startAt: crop.frameTimeS) {
            await PeopleViewer.more(pager) { PeopleViewer.entry($0.crop) }
        }
    }

    private func assign(_ choice: PeopleChoice?) async {
        let ids = Array(selection.selected)
        guard !ids.isEmpty, !busy else { return }
        busy = true
        defer { busy = false }
        let name = self.name
        let r = await PeopleAction.run(choice == nil ? "Took crops off a pet" : "Moved crops", doing: "move the crops") {
            try await Cloud.api().assignAnimals(body: API.AssignAnimalsIn(animalIds: ids, personId: choice?.personId,
                                                                          newPersonName: choice?.newName))
        } message: { r in
            let along = r.followed > 0 ? " \(Fmt.count(r.followed, "look-alike crop")) went along." : ""
            if choice != nil {
                let to = r.person.map { " to \(PeopleText.name($0))" } ?? ""
                return "\(Fmt.count(r.updated, "crop")) moved\(to).\(along)"
            }
            return "\(Fmt.count(r.updated, "crop")) removed from \(name).\(along)"
        }
        if r != nil {
            selection.clear()
            await pager.refresh()
        }
    }

    private func removeBoxes() async {
        let ids = Array(selection.selected)
        guard !ids.isEmpty, !busy else { return }
        busy = true
        defer { busy = false }
        let r = await PeopleAction.run("Removed \(Fmt.count(ids.count, "box", "boxes"))", doing: "remove the boxes") { () async throws -> Int in
            for id in ids { _ = try await Cloud.api().deleteAnimal(animalId: id) }
            return ids.count
        } message: { n in
            "\(Fmt.count(n, "box", "boxes")) removed."
        }
        if r != nil {
            selection.clear()
            await pager.refresh()
        }
    }
}

/// A person's Guesses tab: the guesses waiting for them, each answered on its card, with "Confirm all".
struct PeoplePersonGuessesTab: View {
    let person: API.PersonOut
    let header: AnyView

    @Environment(AppRouter.self) private var router
    @State private var pager: Pager<PeopleGuessRow>
    @State private var busy = false

    init(person: API.PersonOut, header: AnyView) {
        self.person = person
        self.header = header
        let id = person.id
        _pager = State(initialValue: Pager<PeopleGuessRow> { cursor in
            let page = try await Cloud.api().reviewGuesses(cursor: cursor, limit: 60, person: id)
            return .init(items: page.items.map { PeopleGuessRow(guess: $0) }, next: page.nextCursor, total: page.total)
        })
    }

    /// Only this person's guesses (the list could hold another's after a merge).
    private var mine: [PeopleGuessRow] { pager.items.filter { $0.guess.guess.person.id == person.id } }

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                header
                if mine.count > 1 {
                    Button {
                        Task { await confirmAll() }
                    } label: {
                        Label("Confirm all \(mine.count)", systemImage: "checkmark.circle")
                    }
                    .buttonStyle(.bordered)
                    .disabled(busy)
                    .padding(.horizontal, 16)
                    .accessibilityIdentifier("person.guesses.confirmAll")
                }
                if mine.isEmpty {
                    states.padding(.top, 24)
                }
                ForEach(mine) { row in
                    ReviewGuessCard(guess: row.guess, onDecided: {
                        pager.remove(where: { $0.id == row.id })
                    }, onOpenItem: { uid, t in open(uid, at: t) })
                    .padding(.horizontal, 16)
                    .onAppear { pager.loadMoreIfNeeded(row) }
                }
            }
            .padding(.bottom, 16)
        }
        .task { if !pager.loaded { await pager.refresh() } }
        .refreshable { await pager.refresh() }
        .onCloudChange { Task { await pager.refresh() } }
    }

    /// The viewer on the photos of this person's guesses loaded, at `uid` (a video at the box's time when the scene
    /// was tapped); paging past the last one loads more guesses.
    private func open(_ uid: String, at time: Double?) {
        let entries = mine.map(\.guess.item.viewerEntry)
        guard entries.contains(where: { $0.uid == uid }) else {
            router.open(.item(uid: uid))
            return
        }
        let pager = self.pager
        let id = person.id
        PeopleViewer.open(router, entries: entries, start: uid, startAt: time) {
            await PeopleViewer.more(pager) { row in
                row.guess.guess.person.id == id ? row.guess.item.viewerEntry : nil
            }
        }
    }

    @ViewBuilder private var states: some View {
        if let error = pager.error {
            ErrorState(error: error) { Task { await pager.retry() } }
        } else if pager.loaded && !pager.loading {
            EmptyState(title: "No guesses for this person", systemImage: "sparkles",
                       text: "When this person is tagged, other photos within an hour with the same clothes, photos taken moments apart where someone stands in the same spot, and photos taken at any time with the same tattoo, get a guess here. Recognition trained on them adds the faces it finds.")
        } else {
            LoadingState()
        }
    }

    private func confirmAll() async {
        let rows = mine
        guard !busy, !rows.isEmpty else { return }
        let faces = rows.filter { $0.guess.kind == .face }.map(\.guess.id)
        let bodies = rows.filter { $0.guess.kind == .body }.map(\.guess.id)
        busy = true
        defer { busy = false }
        let r = await PeopleAction.run("Confirmed \(Fmt.count(rows.count, "guess", "guesses"))", doing: "save the answers") {
            try await Cloud.api().confirmGuesses(body: API.GuessIdsIn(faces: faces.isEmpty ? nil : faces,
                                                                      bodies: bodies.isEmpty ? nil : bodies))
        } message: { r in
            "\(Fmt.count(r.updated, "guess", "guesses")) confirmed."
        }
        if r != nil { await pager.refresh() }
    }
}
