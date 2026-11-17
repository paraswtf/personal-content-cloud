import SwiftUI

/// One guess in a list (a face or a person box can each have one).
struct ReviewGuessRow: Identifiable {
    let guess: API.GuessItem

    var id: String { "\(guess.kind.rawValue):\(guess.id)" }
}

/// Loads a list's next page for the viewer, waiting first for a page the list itself is loading (its rows'
/// onAppear asked for it), so the viewer does not take a load that returned at once for the end of the list.
@MainActor
enum ReviewPaging {
    static func loadMore<Item: Identifiable>(_ pager: Pager<Item>) async {
        var waited = 0
        while pager.loading && waited < 100 {
            try? await Task.sleep(nanoseconds: 100_000_000)
            waited += 1
        }
        let count = pager.items.count
        await pager.loadMore()
        // the page that was loading may have been the last new one: one more when the list grew by nothing
        if pager.items.count == count && !pager.finished && !pager.loading { await pager.loadMore() }
    }
}

/// What the guesses list shows: every guess, or those of one basis (the website's filter chips) and of one person.
struct ReviewGuessFilter: Hashable {
    var basis: API.Basis = .all
    var person: PersonOption?
}

/// One basis of the website's filter, with its words.
struct ReviewBasisChoice: Identifiable {
    let basis: API.Basis
    let label: String

    var id: String { label }

    static let all: [ReviewBasisChoice] = [
        ReviewBasisChoice(basis: .all, label: "All"), ReviewBasisChoice(basis: .clothes, label: "Clothes"),
        ReviewBasisChoice(basis: .trait, label: "Tattoos and nails"), ReviewBasisChoice(basis: .face, label: "Trained face"),
        ReviewBasisChoice(basis: .place, label: "Position"), ReviewBasisChoice(basis: .group, label: "Grouped"),
    ]
}

/// Guesses (the website's GuessesTab): the cards, the basis filter, a person filter, and Confirm all shown.
struct ReviewGuessesTab: View {
    @Environment(AppRouter.self) private var router
    @State private var filter = ReviewGuessFilter()
    @State private var pager = ReviewGuessesTab.makePager(ReviewGuessFilter())
    @State private var confirmAll = false
    @State private var pickingPerson = false
    @State private var working = false

    static func makePager(_ filter: ReviewGuessFilter) -> Pager<ReviewGuessRow> {
        let basis = filter.basis
        let person = filter.person?.id
        return Pager<ReviewGuessRow> { cursor in
            let page = try await Cloud.api().reviewGuesses(cursor: cursor, limit: 60, person: person, basis: basis)
            return Pager<ReviewGuessRow>.Page(items: page.items.map(ReviewGuessRow.init), next: page.nextCursor,
                                              total: page.total)
        }
    }

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                Text("People guessed from someone tagged within an hour in the same scene and clothes, from standing in the same spot in a photo taken moments apart, from the same tattoo in a photo taken at any time, or by recognition trained on one person. Confirming tags them; rejecting stops that guess.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                filters
                if pager.items.count > 1 {
                    Button {
                        confirmAll = true
                    } label: {
                        Label("Confirm all \(pager.items.count) shown", systemImage: "checkmark.circle")
                    }
                    .buttonStyle(.bordered)
                    .disabled(working)
                    .accessibilityIdentifier("review.confirmAll")
                }
                ForEach(pager.items) { row in
                    ReviewGuessCard(guess: row.guess,
                                    onDecided: { pager.remove { $0.id == row.id } },
                                    onOpenItem: { uid, t in open(uid, at: t) })
                        .onAppear { pager.loadMoreIfNeeded(row) }
                }
                ReviewListState(pager: pager, emptyTitle: emptyTitle, emptyText: emptyText, emptyImage: "sparkles")
            }
            .padding(16)
        }
        .refreshable { await pager.refresh() }
        .confirmAction("Confirm \(Fmt.count(pager.items.count, "guess", "guesses"))?", isPresented: $confirmAll,
                       actionTitle: "Confirm all", role: nil,
                       message: "Every guess shown is tagged as the guessed person. You can still change a tag later from the photo.") {
            Task { await confirmShown() }
        }
        .sheet(isPresented: $pickingPerson) {
            PersonPickerSheet(title: "Guesses of", kind: .person, allowCreate: false) { pick in
                pickingPerson = false
                if case .existing(let option) = pick { filter.person = option }
            }
        }
        .task(id: filter) {
            pager = Self.makePager(filter)
            await pager.refresh()
        }
        .onCloudChange { Task { await pager.refresh() } }
    }

    private var filters: some View {
        ReviewFlowLayout(spacing: 6, lineSpacing: 8) {
                ForEach(ReviewBasisChoice.all) { choice in
                    ToggleChip(text: choice.label,
                               isOn: Binding(get: { filter.basis == choice.basis },
                                             set: { on in if on { filter.basis = choice.basis } }))
                        .accessibilityIdentifier("review.basis.\(choice.basis.rawValue)")
                }
                if let person = filter.person {
                    Button {
                        filter.person = nil
                    } label: {
                        Chip(text: person.name, systemImage: "xmark.circle.fill", tint: .accentColor, filled: true)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Guesses of \(person.name). Show everyone's")
                    .accessibilityIdentifier("review.person.clear")
                } else {
                    Button {
                        pickingPerson = true
                    } label: {
                        Chip(text: "Person", systemImage: "person.crop.circle", tint: .accentColor)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Show the guesses of one person")
                    .accessibilityIdentifier("review.person")
                }
        }
        .padding(.vertical, 2)
    }

    private var emptyTitle: String {
        if let person = filter.person { return "No guesses of \(person.name)" }
        return filter.basis == .all ? "No guesses to review" : "No guesses of this kind"
    }

    private var emptyText: String {
        "Tag someone in a photo: other photos within an hour with the same clothes, photos taken moments apart where someone stands in the same spot, and photos taken at any time with the same tattoo, get a guess. Recognition trained on one person (on their page) adds the faces it finds, and naming a face adds the faces that look like it."
    }

    /// The photos of the guesses loaded, once each, for paging in the viewer.
    private var items: [API.ItemCard] {
        var seen = Set<String>()
        return pager.items.map(\.guess.item).filter { seen.insert($0.uid).inserted }
    }

    /// The viewer on the photos of every guess loaded, at `uid` (a video at the box's time when the scene was
    /// tapped); paging past the last one loads more guesses.
    private func open(_ uid: String, at time: Double?) {
        var list = items.map(\.viewerEntry)
        if let i = list.firstIndex(where: { $0.uid == uid }) {
            if let time, time.isFinite, time > 0 { list[i].startAt = time }
            let pager = self.pager
            router.show(entries: list, start: i) {
                await ReviewPaging.loadMore(pager)
                var seen = Set<String>()
                return pager.items.map(\.guess.item).filter { seen.insert($0.uid).inserted }.map(\.viewerEntry)
            }
        } else {
            router.open(.item(uid: uid))
        }
    }

    private func confirmShown() async {
        let shown = pager.items.map(\.guess)
        guard !shown.isEmpty, !working else { return }
        working = true
        defer { working = false }
        do {
            try await ReviewDecisions.confirm(shown)
            CloudEvents.post(items: Array(Set(shown.map(\.item.uid))))
            await pager.refresh()
        } catch {
            Toaster.shared.error(error, doing: "confirm the guesses")
        }
    }
}
