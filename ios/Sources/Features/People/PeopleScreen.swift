import SwiftUI

/// The People tab (the website's pages/PeoplePage.tsx): the people the owner named, with search, sorting, merge
/// suggestions, picking several to merge, the faces to name, and the pets below. Faces nobody is named for are
/// grouped in the background and never shown as people; they wait in Faces to name.
struct PeopleScreen: View {
    @State private var search = ""
    @State private var query = ""
    @State private var sort: API.ListPeopleSort = .items
    @State private var pager = PeopleScreen.peoplePager(q: "", sort: .items)
    @State private var pets = Pager<PeopleRow> { cursor in
        let page = try await Cloud.api().listPeople(sort: .items, kind: .pet, cursor: cursor, limit: 60)
        return .init(items: page.items.map { PeopleRow(person: $0) }, next: page.nextCursor, total: page.total)
    }
    @State private var picking = SelectionModel<Int>()
    @State private var merging: PeopleMergeRequest?
    @State private var facesToName = false

    static func peoplePager(q: String, sort: API.ListPeopleSort) -> Pager<PeopleRow> {
        Pager<PeopleRow> { cursor in
            let page = try await Cloud.api().listPeople(q: q.isEmpty ? nil : q, sort: sort, kind: .person,
                                                         cursor: cursor, limit: 60)
            // only named people are shown (the server leaves unnamed groups out already)
            let named = page.items.filter { Fmt.isNamed($0.name) }.map { PeopleRow(person: $0) }
            return .init(items: named, next: page.nextCursor, total: page.total)
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text(countText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("people.count")
                if !picking.active && query.isEmpty { PeopleSuggestionsPanel() }
                if picking.active {
                    Text("Pick two or more people who are the same person.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                peopleSection
                if !picking.active { petsSection }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
        }
        .navigationTitle("People")
        .searchable(text: $search, prompt: "Search people by name")
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button {
                    facesToName = true
                } label: {
                    Label("Faces to name", systemImage: "faceid")
                }
                .accessibilityIdentifier("people.facesToName")
            }
            ToolbarItemGroup(placement: .topBarTrailing) {
                Menu {
                    Picker("Sort", selection: $sort) {
                        Text("Most items").tag(API.ListPeopleSort.items)
                        Text("Name").tag(API.ListPeopleSort.name)
                        Text("Recently seen").tag(API.ListPeopleSort.recent)
                    }
                } label: {
                    Label("Sort", systemImage: "arrow.up.arrow.down")
                }
                .accessibilityIdentifier("people.sort")
                Button(picking.active ? "Done" : "Merge") {
                    if picking.active { picking.end() } else { picking.begin() }
                }
                .accessibilityIdentifier("people.merge")
            }
        }
        .safeAreaInset(edge: .bottom) {
            if picking.active && picking.count > 0 { mergeBar }
        }
        .sheet(isPresented: $facesToName) { PeopleFacesToNameSheet() }
        .sheet(item: $merging) { request in
            PeopleMergeSheet(people: request.people) { target, sources in
                Task {
                    if await PeopleMerging.merge(target: target, sources: sources) != nil { picking.end() }
                }
            }
        }
        .task(id: "\(search)|\(sort.rawValue)") { await reload(debounce: query != search.trimmingCharacters(in: .whitespaces)) }
        .refreshable { await refreshAll() }
        .onCloudChange { Task { await refreshAll() } }
    }

    private var countText: String {
        if let total = pager.total { return Fmt.count(total, "person", "people") }
        return pager.loaded ? Fmt.count(pager.items.count, "person", "people") : "Loading"
    }

    @ViewBuilder private var peopleSection: some View {
        if pager.items.isEmpty {
            if let error = pager.error {
                ErrorState(error: error) { Task { await pager.retry() } }
            } else if pager.loaded && !pager.loading {
                EmptyState(title: query.isEmpty ? "No people yet" : "No one matches that name",
                           systemImage: "person.2",
                           text: query.isEmpty ? "Name a face in a photo or in Faces to name, and the person shows here."
                                               : "Try another spelling.")
            } else {
                LoadingState()
            }
        } else {
            PeopleGrid(rows: pager.items, picking: picking, onAppear: { pager.loadMoreIfNeeded($0) })
        }
    }

    @ViewBuilder private var petsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            PeopleSectionHeader(title: "Pets",
                                note: pets.loaded ? Fmt.count(pets.total ?? pets.items.count, "pet") : "Loading")
            if pets.items.isEmpty {
                if let error = pets.error {
                    ErrorState(error: error) { Task { await pets.retry() } }
                } else if pets.loaded {
                    EmptyState(title: "No pets found yet", systemImage: "pawprint",
                               text: "Pets are found in new photos as they are analysed.")
                } else {
                    LoadingState()
                }
            } else {
                PeopleGrid(rows: pets.items, onAppear: { pets.loadMoreIfNeeded($0) })
            }
        }
        .padding(.top, 12)
    }

    private var mergeBar: some View {
        HStack(spacing: 12) {
            Button("Clear") { picking.clear() }
            Spacer()
            Text("\(picking.count) \(picking.count == 1 ? "person" : "people")")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer()
            Button {
                let chosen = pager.items.map(\.person).filter { picking.contains($0.id) }
                if chosen.count >= 2 { merging = PeopleMergeRequest(people: chosen) }
            } label: {
                Label("Merge", systemImage: "arrow.triangle.merge")
            }
            .buttonStyle(.borderedProminent)
            .disabled(picking.count < 2)
            .accessibilityIdentifier("people.merge.start")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private func reload(debounce: Bool) async {
        let q = search.trimmingCharacters(in: .whitespaces)
        if debounce {
            try? await Task.sleep(nanoseconds: 300_000_000)
            if Task.isCancelled { return }
        }
        if q != query || !pager.loaded || pagerSort != sort {
            query = q
            pagerSort = sort
            pager = PeopleScreen.peoplePager(q: q, sort: sort)
            await pager.refresh()
        }
        if !pets.loaded { await pets.refresh() }
    }

    @State private var pagerSort: API.ListPeopleSort = .items

    private func refreshAll() async {
        await pager.refresh()
        await pets.refresh()
    }
}
