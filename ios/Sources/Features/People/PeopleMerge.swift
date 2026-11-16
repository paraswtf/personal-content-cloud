import SwiftUI

/// Choose which person to keep when merging two or more (the website's MergeModal).
struct PeopleMergeSheet: View {
    let people: [API.PersonOut]
    let onConfirm: (_ target: Int, _ sources: [Int]) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var target: Int

    init(people: [API.PersonOut], onConfirm: @escaping (_ target: Int, _ sources: [Int]) -> Void) {
        self.people = people
        self.onConfirm = onConfirm
        // a named person first, then the one with more items, then the older one
        let ranked = people.sorted { a, b in
            let na = Fmt.isNamed(a.name), nb = Fmt.isNamed(b.name)
            if na != nb { return na }
            if a.itemCount != b.itemCount { return a.itemCount > b.itemCount }
            return a.id < b.id
        }
        _target = State(initialValue: ranked.first?.id ?? 0)
    }

    private var keep: API.PersonOut? { people.first { $0.id == target } ?? people.first }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(people, id: \.id) { p in
                        Button {
                            target = p.id
                        } label: {
                            HStack(spacing: 12) {
                                Avatar(name: p.name, ref: p.coverUrl, size: 44)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(PeopleText.name(p)).font(.subheadline.weight(.semibold))
                                    Text("\(Fmt.count(p.itemCount, "item")), \(Fmt.count(p.faceCount, "face"))")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if target == p.id {
                                    Text("Keep").font(.caption.weight(.semibold)).foregroundStyle(Color.accentColor)
                                    Image(systemName: "checkmark").foregroundStyle(Color.accentColor)
                                }
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityAddTraits(target == p.id ? .isSelected : [])
                    }
                } header: {
                    Text("Choose the person to keep. All faces, bodies and tags move to them. If they have no name, they take the name of one of the others.")
                        .textCase(nil)
                }
            }
            .navigationTitle("Merge \(people.count) people")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Merge") {
                        guard let keep else { return }
                        let sources = people.map(\.id).filter { $0 != keep.id }
                        dismiss()
                        onConfirm(keep.id, sources)
                    }
                    .accessibilityLabel(keep.map { "Merge into \(PeopleText.name($0))" } ?? "Merge")
                    .accessibilityIdentifier("people.merge.confirm")
                }
            }
        }
    }
}

/// Merges people and says so: "Merged into Asha." with the look-alikes that came along.
@MainActor
enum PeopleMerging {
    @discardableResult
    static func merge(target: Int, sources: [Int], withTail: Bool = true) async -> API.PersonOut? {
        await PeopleAction.run("Merged people", doing: "merge") {
            try await Cloud.api().mergePeople(body: API.MergeIn(targetId: target, sourceIds: sources))
        } message: { p in
            "Merged into \(PeopleText.name(p)).\(withTail ? Fmt.followedTail(p.followed) : "")"
        }
    }
}

/// Merge suggestions (the website's SuggestionsPanel): pairs of named people who look alike, with "Same person"
/// and "Different". A childhood pair is two people whose childhood photos look alike: "Same child?".
struct PeopleSuggestionsPanel: View {
    @State private var items: [API.Suggestion] = []
    @State private var choose: PeopleMergeRequest?
    @State private var busyPair: String?

    var body: some View {
        Group {
            if !items.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    PeopleSectionHeader(title: "Could these be the same person?", systemImage: "person.2")
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(alignment: .top, spacing: 12) {
                            ForEach(items, id: \.self) { s in card(s) }
                        }
                        .padding(.vertical, 2)
                    }
                    .accessibilityIdentifier("people.suggestions")
                }
            }
        }
        .task { await load() }
        .onCloudChange { Task { await load() } }
        .sheet(item: $choose) { request in
            PeopleMergeSheet(people: request.people) { target, sources in
                Task { await PeopleMerging.merge(target: target, sources: sources, withTail: false) }
            }
        }
    }

    private func key(_ s: API.Suggestion) -> String { "\(s.a.id)-\(s.b.id)" }

    private func card(_ s: API.Suggestion) -> some View {
        let busy = busyPair == key(s)
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Spacer()
                if s.kind == .childhood {
                    Chip(text: "Same child?")
                } else {
                    Chip(text: "\(Int((s.similarity * 100).rounded()))% alike", tint: s.similarity >= 0.6 ? .green : .secondary)
                }
                Spacer()
            }
            HStack(alignment: .top, spacing: 12) {
                side(s.a, samples: s.aSamples)
                side(s.b, samples: s.bSamples)
            }
            if s.kind == .childhood, Fmt.isNamed(s.a.name) || Fmt.isNamed(s.b.name) {
                Text("Compared with the childhood photos already on \(PeopleText.name(Fmt.isNamed(s.a.name) ? s.a : s.b)).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                Button {
                    startMerge(s)
                } label: {
                    Label("Same person", systemImage: "arrow.triangle.merge").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier("people.suggestion.same")
                Button {
                    Task { await dismissPair(s) }
                } label: {
                    Text("Different").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("people.suggestion.different")
            }
            .disabled(busy)
            .controlSize(.small)
        }
        .padding(12)
        .frame(width: 300)
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(uiColor: .secondarySystemBackground)))
    }

    private func side(_ p: API.PersonOut, samples: [String]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            NavigationLink(value: Route.person(id: p.id)) {
                HStack(spacing: 8) {
                    Avatar(name: p.name, ref: p.coverUrl, size: 40)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(PeopleText.name(p)).font(.subheadline.weight(.semibold)).lineLimit(1)
                        Text(Fmt.count(p.itemCount, "item")).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .buttonStyle(.plain)
            .accessibilityLabel("\(PeopleText.name(p)), \(Fmt.count(p.itemCount, "item"))")
            HStack(spacing: 3) {
                ForEach(Array(samples.prefix(4).enumerated()), id: \.offset) { _, ref in
                    RemoteImage(ref: ref, maxPixel: 128)
                        .frame(width: 30, height: 30)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }
            .accessibilityHidden(true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func load() async {
        do {
            items = try await Cloud.api().suggestions(limit: 12).items
        } catch {
            // the panel only shows when there is something to suggest
        }
    }

    private func startMerge(_ s: API.Suggestion) {
        if Fmt.isNamed(s.a.name) && Fmt.isNamed(s.b.name) {
            choose = PeopleMergeRequest(people: [s.a, s.b])
            return
        }
        let keepA = Fmt.isNamed(s.a.name) ? true : Fmt.isNamed(s.b.name) ? false : s.a.itemCount >= s.b.itemCount
        let (t, src) = keepA ? (s.a, s.b) : (s.b, s.a)
        busyPair = key(s)
        Task {
            await PeopleMerging.merge(target: t.id, sources: [src.id], withTail: false)
            busyPair = nil
        }
    }

    private func dismissPair(_ s: API.Suggestion) async {
        busyPair = key(s)
        defer { busyPair = nil }
        await PeopleAction.run("Marked as different people", doing: "save that") {
            try await Cloud.api().dismissSuggestion(body: API.DismissIn(a: s.a.id, b: s.b.id))
        } message: { _ in
            "Marked as different people."
        }
    }
}
