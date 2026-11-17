import SwiftUI

/// The library's filter controls (the website's components/library/FilterBar.tsx): the category chips (not in the
/// vault), Filters for the rest in a sheet (people, tags, type, dates, favourites, Live Photos, mis-clicks,
/// forwards, every copy, sort), and Clear filters. The map uses it with its own endpoints; `searching` leaves out
/// what a search does not take.
struct LibraryFilterBar: View {
    @Binding var filters: LibraryFilters
    var scope: LibraryScope = .all
    var searching = false
    /// Given by a screen that presents the filter sheet itself, so the sheet stays open while the list under it
    /// changes; without it the bar presents the sheet.
    var presentFilters: (() -> Void)? = nil

    @State private var showingFilters = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if !scope.isVault {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            LibraryChip(title: "All", isOn: filters.category == nil) { filters.category = nil }
                                .accessibilityIdentifier("library.category.all")
                            ForEach(LibraryFilters.categories) { choice in
                                LibraryChip(title: choice.label, isOn: filters.category == choice.value) {
                                    filters.category = filters.category == choice.value ? nil : choice.value
                                }
                                .accessibilityIdentifier("library.category.\(choice.value.rawValue)")
                            }
                        }
                        .padding(.vertical, 2)
                    }
                } else {
                    Spacer(minLength: 0)
                }
                Button {
                    if let presentFilters { presentFilters() } else { showingFilters = true }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "slider.horizontal.3")
                        Text("Filters")
                        if filters.hasExtra {
                            Circle().fill(Color.accentColor).frame(width: 7, height: 7)
                        }
                    }
                    .font(.subheadline.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .accessibilityLabel(filters.hasExtra ? "Filters, some filters are on" : "Filters")
                .accessibilityIdentifier("library.filters")
            }
            if filters.any {
                Button {
                    filters = LibraryFilters()
                } label: {
                    Label("Clear filters", systemImage: "xmark")
                        .font(.subheadline.weight(.semibold))
                }
                .accessibilityIdentifier("library.clearFilters")
            }
        }
        .sheet(isPresented: $showingFilters) {
            LibraryFilterSheet(filters: $filters, scope: scope, searching: searching)
        }
    }
}

/// A round chip that is on or off.
struct LibraryChip: View {
    let title: String
    let isOn: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .foregroundStyle(isOn ? Color(uiColor: .systemBackground) : Color.primary)
                .background(isOn ? Color.primary : Color(uiColor: .secondarySystemFill), in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}

/// The filters beyond the categories, in a sheet; every change applies at once.
struct LibraryFilterSheet: View {
    @Binding var filters: LibraryFilters
    let scope: LibraryScope
    let searching: Bool

    @Environment(\.dismiss) private var dismiss
    @State private var names: [Int: String] = [:]
    @State private var picking = false
    @State private var tags: [API.TagOut] = []
    @State private var showAllTags = false

    /// Tags shown before "Show all": the most used first (the website's TAGS_SHOWN).
    private static let tagsShown = 12

    var body: some View {
        NavigationStack {
            Form {
                peopleSection
                tagsSection
                Section("Type") {
                    Picker("Type", selection: kindBinding) {
                        ForEach(LibraryFilters.kinds) { choice in
                            Text(choice.label).tag(choice.value)
                        }
                    }
                    .accessibilityIdentifier("library.filter.kind")
                }
                Section("Dates") {
                    Toggle("From", isOn: dayOn(\.from))
                        .accessibilityIdentifier("library.filter.fromOn")
                    if filters.from != nil {
                        DatePicker("From", selection: dayValue(\.from),
                                   in: ...(LibraryFilters.day(filters.to) ?? Date.distantFuture),
                                   displayedComponents: .date)
                    }
                    Toggle("To", isOn: dayOn(\.to))
                        .accessibilityIdentifier("library.filter.toOn")
                    if filters.to != nil {
                        DatePicker("To", selection: dayValue(\.to),
                                   in: (LibraryFilters.day(filters.from) ?? Date.distantPast)...,
                                   displayedComponents: .date)
                    }
                }
                Section {
                    Toggle(isOn: flag(\.favorite)) { Label("Favourites", systemImage: "star") }
                        .accessibilityIdentifier("library.filter.favorite")
                    Toggle(isOn: flag(\.live)) { Label("Live Photos", systemImage: "livephoto") }
                        .accessibilityIdentifier("library.filter.live")
                    if !searching {
                        Toggle(isOn: included(\.misclicks)) {
                            Label("Include mis-clicks", systemImage: "exclamationmark.triangle")
                        }
                        .accessibilityIdentifier("library.filter.misclicks")
                        Toggle(isOn: included(\.forwards)) {
                            Label("Include forwards", systemImage: "arrowshape.turn.up.right")
                        }
                        .accessibilityIdentifier("library.filter.forwards")
                        Toggle(isOn: everyCopy) { Label("Show every copy", systemImage: "square.on.square") }
                            .accessibilityIdentifier("library.filter.dups")
                    }
                }
                if !searching {
                    Section("Sort") {
                        Picker("Sort", selection: sortBinding) {
                            Text("Date taken").tag(API.Sort.taken)
                            Text("Date added").tag(API.Sort.added)
                        }
                        .pickerStyle(.segmented)
                        .accessibilityIdentifier("library.filter.sort")
                    }
                }
                if filters.any {
                    Section {
                        Button("Clear filters", role: .destructive) { filters = LibraryFilters() }
                            .accessibilityIdentifier("library.filter.clear")
                    }
                }
            }
            .navigationTitle("Filters")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .accessibilityIdentifier("library.filters.done")
                }
            }
            .sheet(isPresented: $picking) {
                PersonPickerSheet(title: "Add a person or pet", kind: .all, excluding: Set(filters.person ?? []),
                                  allowCreate: false) { pick in
                    if case .existing(let option) = pick {
                        names[option.id] = option.name
                        var people = filters.person ?? []
                        if !people.contains(option.id) { people.append(option.id) }
                        filters.person = people
                    }
                }
            }
            .task { await load() }
        }
    }

    // MARK: People

    private var peopleSection: some View {
        Section("People") {
            ForEach(filters.person ?? [], id: \.self) { id in
                HStack {
                    Text(names[id] ?? "Loading")
                    Spacer()
                    Button {
                        filters.person = (filters.person ?? []).filter { $0 != id }
                    } label: {
                        Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Remove \(names[id] ?? "this person")")
                }
            }
            Button((filters.person ?? []).isEmpty ? "Add a person or pet" : "Add another person or pet") {
                picking = true
            }
            .accessibilityIdentifier("library.filter.addPerson")
        }
    }

    // MARK: Tags

    /// The tags with items, and the chosen ones whatever their count.
    private var usableTags: [API.TagOut] {
        let chosen = filters.tag ?? []
        return tags.filter { t in
            t.items + t.autoItems > 0 || chosen.contains(where: { $0.caseInsensitiveCompare(t.name) == .orderedSame })
        }
    }

    @ViewBuilder
    private var tagsSection: some View {
        let usable = usableTags
        let chosen = filters.tag ?? []
        if !usable.isEmpty || !chosen.isEmpty {
            let shown = showAllTags ? usable : Array(usable.prefix(Self.tagsShown))
            // chosen tags outside the shown ones (further down, renamed or deleted) stay listed to be removed
            let missing = chosen.filter { name in
                !shown.contains(where: { $0.name.caseInsensitiveCompare(name) == .orderedSame })
            }
            Section("Tags") {
                ForEach(missing, id: \.self) { name in
                    tagRow(name: name, detail: nil)
                }
                ForEach(shown, id: \.id) { tag in
                    tagRow(name: tag.name, detail: tag.autoItems > 0
                           ? "\(tag.items.formatted()) tagged by you, \(tag.autoItems.formatted()) suggested"
                           : "\(tag.items.formatted()) tagged by you")
                }
                if usable.count > Self.tagsShown {
                    Button(showAllTags ? "Show fewer tags" : "Show all \(usable.count) tags") {
                        showAllTags.toggle()
                    }
                }
            }
        }
    }

    private func tagRow(name: String, detail: String?) -> some View {
        let isOn = (filters.tag ?? []).contains(where: { $0.caseInsensitiveCompare(name) == .orderedSame })
        return Button {
            var list = filters.tag ?? []
            if isOn {
                list.removeAll { $0.caseInsensitiveCompare(name) == .orderedSame }
            } else {
                list.append(name)
            }
            filters.tag = LibraryFilters.cleanTags(list)
        } label: {
            HStack {
                Image(systemName: "tag")
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(name).foregroundStyle(.primary)
                    if let detail {
                        Text(detail).font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                if isOn {
                    Image(systemName: "checkmark").foregroundStyle(Color.accentColor)
                }
            }
        }
        .accessibilityAddTraits(isOn ? .isSelected : [])
        .accessibilityIdentifier("library.filter.tag")
    }

    // MARK: Bindings

    private var kindBinding: Binding<API.Kind?> {
        Binding(get: { filters.kind }, set: { filters.kind = $0 })
    }

    private var sortBinding: Binding<API.Sort> {
        Binding(get: { filters.sort ?? .taken }, set: { filters.sort = $0 == .added ? .added : nil })
    }

    private var everyCopy: Binding<Bool> {
        Binding(get: { filters.dups == .all }, set: { filters.dups = $0 ? .all : nil })
    }

    private func flag(_ key: WritableKeyPath<LibraryFilters, Bool?>) -> Binding<Bool> {
        Binding(get: { filters[keyPath: key] == true }, set: { filters[keyPath: key] = $0 ? true : nil })
    }

    private func included(_ key: WritableKeyPath<LibraryFilters, API.Forwards?>) -> Binding<Bool> {
        Binding(get: {
            if let value = filters[keyPath: key] { return value != .exclude }
            return false
        }, set: { filters[keyPath: key] = $0 ? .include : nil })
    }

    private func dayOn(_ key: WritableKeyPath<LibraryFilters, String?>) -> Binding<Bool> {
        Binding(get: { filters[keyPath: key] != nil },
                set: { filters[keyPath: key] = $0 ? LibraryFilters.dayString(Date()) : nil })
    }

    private func dayValue(_ key: WritableKeyPath<LibraryFilters, String?>) -> Binding<Date> {
        Binding(get: { LibraryFilters.day(filters[keyPath: key]) ?? Date() },
                set: { filters[keyPath: key] = LibraryFilters.dayString($0) })
    }

    // MARK: Loading

    private func load() async {
        if let list = try? await Cloud.api().listTags() {
            tags = list.sorted { ($0.items + $0.autoItems) > ($1.items + $1.autoItems) }
        }
        for id in filters.person ?? [] where names[id] == nil {
            if let person = try? await Cloud.api().getPerson(personId: id) {
                names[id] = Fmt.personName(person.name, kind: person.kind.rawValue, species: person.species)
            }
        }
    }
}
