import SwiftUI

/// The faces to name (the website's UnassignedFaces): faces of nobody and faces of the groups the app keeps in the
/// background, a group's faces side by side. Pick some and tag them (the other faces of their groups then wait in
/// Review as guesses), or remove them as not faces. Match again runs the automatic matching.
struct PeopleFacesToNameSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var pager = Pager<PeopleFaceRow> { cursor in
        let page = try await Cloud.api().unassignedFaces(cursor: cursor, limit: 120)
        return .init(items: page.items.map { PeopleFaceRow(face: $0) }, next: page.nextCursor, total: page.total)
    }
    @State private var selection = SelectionModel<Int>()
    @State private var tagging = false
    @State private var busy = false

    var body: some View {
        NavigationStack {
            ScrollView {
                if pager.items.isEmpty {
                    states.padding(.top, 40)
                } else {
                    PeopleFaceGrid(faces: pager.items, selection: selection,
                                   onAppear: { pager.loadMoreIfNeeded($0) })
                        .padding(.horizontal, 12)
                        .padding(.bottom, 12)
                }
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                        .accessibilityIdentifier("faces.done")
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        Task { await matchAgain() }
                    } label: {
                        Label("Match again", systemImage: "arrow.clockwise")
                    }
                    .disabled(busy)
                    .accessibilityIdentifier("faces.matchAgain")
                }
            }
            .safeAreaInset(edge: .bottom) { bar }
            .sheet(isPresented: $tagging) {
                PersonPickerSheet(title: "Tag \(Fmt.count(selection.count, "face")) as",
                                  like: .faces(Array(selection.selected))) { pick in
                    Task { await tag(PeopleChoice(pick)) }
                }
            }
            .task {
                // the drawer is always picking: a tap picks a face
                if !selection.active { selection.begin() }
                if !pager.loaded { await pager.refresh() }
            }
            .refreshable { await pager.refresh() }
        }
        .toastHost()
    }

    private var title: String {
        if let total = pager.total, total > 0 { return "Faces to name (\(total.formatted()))" }
        return "Faces to name"
    }

    @ViewBuilder private var states: some View {
        if let error = pager.error {
            ErrorState(error: error) { Task { await pager.retry() } }
        } else if pager.loaded && !pager.loading {
            EmptyState(title: "Every face has a name", systemImage: "face.smiling",
                       text: "New faces that match nobody you named show up here.")
        } else {
            LoadingState()
        }
    }

    private var bar: some View {
        HStack(spacing: 12) {
            Text(selection.count > 0 ? "\(selection.count) selected" : "Select faces to tag them")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("faces.count")
            Spacer()
            Button(role: .destructive) {
                Task { await notFaces() }
            } label: {
                Label("Not faces", systemImage: "nosign")
            }
            .disabled(selection.count == 0 || busy)
            .accessibilityIdentifier("faces.notFaces")
            Button {
                tagging = true
            } label: {
                Label("Tag as", systemImage: "person.badge.plus")
            }
            .buttonStyle(.borderedProminent)
            .disabled(selection.count == 0 || busy)
            .accessibilityIdentifier("faces.tag")
        }
        .labelStyle(.titleAndIcon)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private func tag(_ choice: PeopleChoice) async {
        let ids = Array(selection.selected)
        guard !ids.isEmpty else { return }
        busy = true
        defer { busy = false }
        let answer = await PeopleAction.run("Tagged \(Fmt.count(ids.count, "face"))", doing: "assign the faces") {
            try await Cloud.api().assignFaces(body: API.AssignFacesIn(faceIds: ids, personId: choice.personId,
                                                                      newPersonName: choice.newName))
        } message: { r in
            let who = r.person.map { " as \(PeopleText.name($0))" } ?? ""
            return "\(Fmt.count(r.updated, "face")) tagged\(who).\(Fmt.guessedTail(r.guessed))"
        }
        if answer != nil {
            selection.clear()
            await pager.refresh()
        }
    }

    private func notFaces() async {
        let ids = Array(selection.selected)
        guard !ids.isEmpty else { return }
        busy = true
        defer { busy = false }
        var failed = 0
        var firstError: Error?
        let done = await PeopleAction.run("Removed \(Fmt.count(ids.count, "face")) as not faces", doing: "remove the faces") { () async throws -> Int in
            var removed = 0
            for id in ids {
                do {
                    _ = try await Cloud.api().deleteFace(faceId: id)
                    removed += 1
                } catch {
                    failed += 1
                    if firstError == nil { firstError = error }
                }
            }
            return removed
        } message: { removed in
            failed > 0 ? nil : "\(Fmt.count(removed, "face")) removed and kept as examples of what is not a face."
        }
        if failed > 0, let firstError {
            Toaster.shared.error("\(failed) could not be removed. \(ErrorText.describe(firstError))")
        }
        if done != nil {
            selection.clear()
            await pager.refresh()
        }
    }

    private func matchAgain() async {
        busy = true
        defer { busy = false }
        await PeopleAction.run("Matched faces again", doing: "start matching") {
            try await Cloud.api().recluster()
        } message: { _ in
            "Matching faces again. Faces that match someone you named leave this list in a minute."
        }
    }
}
