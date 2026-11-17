import SwiftUI

/// Recognition trained on one person (the website's RecognitionCard, docs/ARCHITECTURE.md section 10), on that
/// person's page. "Train and look again" fits a model on the faces now on NAME (and on the faces taken off NAME as
/// wrong), checks it on photos it was not trained on, then looks at every other face and adds what looks like NAME
/// as guesses. Nothing is tagged until the owner confirms. The card follows a run's progress; two lists help clean up:
/// faces on NAME that look like faces taken off NAME, and faces on other people that look like NAME.
struct PeopleRecognitionCard: View {
    let person: API.PersonOut
    /// The person page loads the status with the person (a view that shows nothing until it has loaded would never
    /// start loading); the card follows a run from there and hands every newer status back with `onStatus`.
    let status: API.RecognitionOut
    let onStatus: (API.RecognitionOut) -> Void

    /// Whether the card shows at all: nothing when it cannot be trained yet and never was (too few faces, learning
    /// off), as on the website.
    static func shows(_ s: API.RecognitionOut) -> Bool {
        s.trained || s.available || s.running != nil || s.error != nil
    }

    private enum Shown { case conflicts, elsewhere }
    private enum Asking: Identifiable {
        case withdraw, forget
        var id: Int { self == .withdraw ? 0 : 1 }
    }

    @State private var shown: Shown?
    @State private var asking: Asking?
    @State private var starting = false
    @State private var pollToken = 0
    @State private var wasRunning: API.Running?

    private var name: String { PeopleText.name(person) }

    var body: some View {
        card(status)
        .task(id: "\(person.id)-\(pollToken)") { await poll() }
        .confirmationDialog(askTitle, isPresented: Binding(get: { asking != nil }, set: { if !$0 { asking = nil } }),
                            titleVisibility: .visible, presenting: asking) { kind in
            switch kind {
            case .withdraw:
                Button("Take them back") { Task { await withdraw() } }
            case .forget:
                Button("Forget", role: .destructive) { Task { await forget() } }
            }
            Button("Cancel", role: .cancel) {}
        } message: { kind in
            switch kind {
            case .withdraw:
                Text("The \(Fmt.count(status.guesses, "guess", "guesses")) this recognition made go away, and it makes no new ones until you look again. Faces you already confirmed stay.")
            case .forget:
                Text("Its open guesses go away and faces are matched as before the training. The faces and your answers stay, so you can train again at any time.")
            }
        }
    }

    private var askTitle: String {
        switch asking {
        case .withdraw: return "Take back the guesses for \(name)?"
        case .forget: return "Forget the recognition trained on \(name)?"
        case .none: return ""
        }
    }

    // MARK: the card

    private func card(_ s: API.RecognitionOut) -> some View {
        let busy = s.running != nil || starting || s.busy
        return VStack(alignment: .leading, spacing: 6) {
            Label(s.trained ? "Recognition trained on \(name)" : "Train recognition on \(name) alone",
                  systemImage: "faceid")
                .font(.subheadline.weight(.semibold))
            if let running = s.running {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(stepText(s, running: running)).font(.caption).foregroundStyle(.secondary)
                }
                if let p = s.progress, p.total > 0 {
                    ProgressView(value: Double(p.done), total: Double(p.total))
                        .accessibilityLabel(running == .scan ? "Looking at other faces" : "Training")
                } else {
                    ProgressView().progressViewStyle(.linear)
                        .accessibilityLabel(running == .scan ? "Looking at other faces" : "Training")
                }
            } else if !s.trained {
                note("It takes the \(Fmt.count(person.faceCount, "face")) now on \(name) as right and the faces you took off \(name) as wrong, so take wrong faces off first. Then it looks again at every other face and adds what looks like \(name) as guesses. Nothing is tagged until you confirm.")
                if let error = s.error { note(error, tint: .red) }
                if !s.available, let reason = s.reason { note(reason) }
                if s.busy { note("Another person's training is running. This one can start when it is done.") }
                Button("Train and look again") { Task { await start(.train) } }
                    .buttonStyle(.bordered)
                    .disabled(!s.available || busy)
                    .padding(.top, 4)
                    .accessibilityIdentifier("person.recognition.train")
            } else {
                note(trainedText(s))
                if let scan = s.scan {
                    let waiting = s.guesses > 0 ? "\(Fmt.count(s.guesses, "guess", "guesses")) waiting for you" : "no guess waiting"
                    let none = scan.guesses == 0 ? ". It found no face that looks like \(name) and is not tagged yet." : "."
                    note("Last look, \(Fmt.date(scan.at, withTime: true)): \(Fmt.count(scan.looked, "other face")), \(waiting)\(none)")
                }
                if s.withdrawn { note("Its guesses were taken back. It makes no new ones until you look again.") }
                if !s.available, let reason = s.reason {
                    note(reason + (s.learning ? "" : " The trained recognition is not used meanwhile."))
                }
                if let error = s.error { note(error, tint: .red) }
                if s.busy { note("Another person's training is running. This one can run when it is done.") }
                if s.answersSince > 0 {
                    note("\(Fmt.count(s.answersSince, "answer")) about \(name) since the training. Train again to use them.")
                }
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        Button("Look again") { Task { await start(.scan) } }
                            .disabled(!s.learning || busy)
                        Button("Train again") { Task { await start(.train) } }
                            .disabled(!s.available || busy)
                        if s.guesses > 0 {
                            Button("Take back its guesses") { asking = .withdraw }.disabled(busy)
                        }
                        Button("Forget", role: .destructive) { asking = .forget }.disabled(busy)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                .padding(.top, 4)
                if s.conflicts > 0 {
                    Divider().padding(.vertical, 4)
                    note("\(Fmt.count(s.conflicts, "face")) on \(name) \(s.conflicts == 1 ? "looks" : "look") the same as faces you took off \(name). Settle \(s.conflicts == 1 ? "it" : "these") and train again.",
                         tint: .primary)
                    Button(shown == .conflicts ? "Hide them" : "Show them") {
                        shown = shown == .conflicts ? nil : .conflicts
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    if shown == .conflicts { PeopleRecognitionConflicts(person: person) }
                }
                if s.elsewhere > 0 {
                    Divider().padding(.vertical, 4)
                    note("\(Fmt.count(s.elsewhere, "face")) the app put on other people \(s.elsewhere == 1 ? "looks" : "look") like \(name).",
                         tint: .primary)
                    Button(shown == .elsewhere ? "Hide them" : "Show them") {
                        shown = shown == .elsewhere ? nil : .elsewhere
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    if shown == .elsewhere { PeopleRecognitionElsewhere(person: person) }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(uiColor: .secondarySystemBackground)))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Recognition trained on \(name)")
    }

    private func note(_ text: String, tint: Color = .secondary) -> some View {
        Text(text).font(.caption).foregroundStyle(tint).fixedSize(horizontal: false, vertical: true)
    }

    /// What a run is doing, in words.
    private func stepText(_ s: API.RecognitionOut, running: API.Running) -> String {
        let p = s.progress
        if running == .scan {
            if p?.step == "write" { return "Writing the guesses." }
            if let p, p.total > 0 { return "Looking at other faces: \(p.done.formatted()) of \(p.total.formatted())." }
            return "Looking at other faces."
        }
        switch p?.step {
        case "check": return "Checking it on photos it was not trained on."
        case "save": return "Saving."
        case "fit":
            if let p { return "Training: step \(min(p.done + 1, p.total)) of \(p.total)." }
            return "Training."
        default: return "Reading the faces."
        }
    }

    private func trainedText(_ s: API.RecognitionOut) -> String {
        let positives = Int(s.counts["positives"] ?? 0), negatives = Int(s.counts["negatives"] ?? 0)
        var text = "Trained \(Fmt.date(s.trainedAt, withTime: true)) on \(Fmt.count(positives, "face")) of \(name), against \(Fmt.count(negatives, "face")) known not to be \(name)."
        if let check = s.check, let cut = check.cuts.first {
            let held = check.folds == .day ? "On days it was not trained on" : "On photos it was not trained on"
            text += " \(held), it found \(cut.found.formatted()) of \(name)'s \(check.positives.formatted()) faces and mistook \(cut.mistaken.formatted()) of \(check.negatives.formatted()) faces of other people for \(name)."
            if check.unclearPositives > 0 {
                text += " Of the \(check.unclearPositives.formatted()) blurred, small or turned ones it found \(cut.unclearFound.formatted())."
            }
        }
        return text
    }

    // MARK: loading and actions

    /// Follows a run (every 2 s) or another person's run (every 6 s) until it ends.
    private func poll() async {
        var current = status
        if let running = current.running, wasRunning == nil { wasRunning = running }
        if pollToken > 0, let fresh = await load() { current = fresh }
        while !Task.isCancelled, current.running != nil || current.busy {
            try? await Task.sleep(nanoseconds: current.running != nil ? 2_000_000_000 : 6_000_000_000)
            if Task.isCancelled { return }
            if let fresh = await load() { current = fresh }
        }
    }

    @discardableResult
    private func load() async -> API.RecognitionOut? {
        guard let s = try? await Cloud.api().personRecognition(personId: person.id) else { return nil }
        // a run that ends: say what it did, and let the Guesses tab and the counts catch up
        if let was = wasRunning, s.running == nil {
            let waiting = Fmt.count(s.guesses, "guess", "guesses")
            if let error = s.error {
                Toaster.shared.error(error)
            } else if was == .train {
                Toaster.shared.ok("Recognition trained on \(name): \(waiting) to review.")
            } else {
                Toaster.shared.ok("Looked again for \(name): \(waiting) to review.")
            }
            wasRunning = nil
            onStatus(s)
            CloudEvents.post()
            return s
        }
        if let running = s.running, wasRunning == nil { wasRunning = running }
        onStatus(s)
        return s
    }

    private func startMessage(_ error: Error) -> String? {
        guard let api = error as? APIError else { return nil }
        switch api.detail {
        case "busy": return "Another person's training is running. Try again when it is done."
        case "running": return "A training or a look for \(name) is running. Try again when it is done."
        case "not_trained": return "Train it first."
        default: return nil
        }
    }

    private func start(_ kind: API.Running) async {
        starting = true
        defer { starting = false }
        do {
            let r: API.RecognitionStartOut
            if kind == .train {
                r = try await UndoCenter.shared.perform("Trained recognition", toast: false) {
                    try await Cloud.api().trainRecognition(personId: person.id, body: API.RecognitionStartIn(scan: true))
                }
            } else {
                r = try await UndoCenter.shared.perform("Looked again", toast: false) {
                    try await Cloud.api().scanRecognition(personId: person.id)
                }
            }
            if !r.started {
                Toaster.shared.ok("A training is already running.")
            } else {
                wasRunning = kind
            }
            pollToken += 1
        } catch {
            if let text = startMessage(error) {
                Toaster.shared.error(text)
            } else {
                Toaster.shared.error(error, doing: kind == .train ? "start the training" : "start looking")
            }
        }
    }

    private func withdraw() async {
        let name = self.name
        await PeopleAction.run("Took back recognition guesses", doing: "take the guesses back") {
            try await Cloud.api().withdrawRecognition(personId: person.id)
        } message: { r in
            "Took back \(Fmt.count(r.updated, "guess", "guesses")) of the recognition trained on \(name)."
        }
        await load()
    }

    private func forget() async {
        let name = self.name
        let r = await PeopleAction.run("Forgot a recognition", doing: "forget it") {
            try await Cloud.api().forgetRecognition(personId: person.id)
        } message: { _ in
            "Forgot the recognition trained on \(name)."
        }
        if r != nil { shown = nil }
        await load()
    }
}

/// One face answered through the usual assignment: to the person, to someone else, or to nobody.
@MainActor
private func peopleSettle(face: Int, person: Int?, text: String) async {
    await PeopleAction.run(text, doing: "save the answer") {
        try await Cloud.api().assignFaces(body: API.AssignFacesIn(faceIds: [face], personId: person))
    } message: { r in
        "\(text)\(Fmt.followedTail(r.followed))"
    }
}

/// A face crop that opens its photo.
private struct PeopleRecognitionCrop: View {
    let face: API.FaceListItem
    let label: String

    @Environment(AppRouter.self) private var router

    var body: some View {
        Button {
            router.open(.item(uid: face.itemUid))
        } label: {
            RemoteImage(ref: face.url, maxPixel: 192)
                .frame(width: 56, height: 56)
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

/// Faces on the person that look like faces the owner took off them: keep, take off, or move to the other one.
struct PeopleRecognitionConflicts: View {
    let person: API.PersonOut

    @State private var page: API.RecognitionConflictPage?
    @State private var loadError: Error?

    private var name: String { PeopleText.name(person) }

    var body: some View {
        Group {
            if let page {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(page.items, id: \.face.id) { c in row(c) }
                    if page.total > page.items.count {
                        Text("Showing \(page.items.count) of \(page.total.formatted()). The rest shows as you settle these.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } else if let loadError {
                Text(ErrorText.describe(loadError, doing: "load them")).font(.caption).foregroundStyle(.red)
            } else {
                Text("Loading.").font(.caption).foregroundStyle(.secondary)
            }
        }
        .task { await load() }
        .onCloudChange { Task { await load() } }
    }

    private func row(_ c: API.RecognitionConflict) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                PeopleRecognitionCrop(face: c.face, label: "Open the photo of the face on \(name)")
                Text("looks like").font(.caption).foregroundStyle(.secondary)
                if let like = c.like {
                    PeopleRecognitionCrop(face: like, label: "Open the photo of the face it looks like")
                } else {
                    Text("a face that cannot be shown").font(.caption).foregroundStyle(.secondary)
                }
                Text(c.likePerson.map { "now on \(Fmt.personName($0.name))" } ?? "which you took off")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    Button("This is \(name)") {
                        Task { await peopleSettle(face: c.face.id, person: person.id, text: "Kept on \(name).") }
                    }
                    Button("Not \(name)") {
                        Task { await peopleSettle(face: c.face.id, person: nil, text: "Taken off \(name).") }
                    }
                    if let other = c.likePerson {
                        Button("Move to \(Fmt.personName(other.name))") {
                            Task { await peopleSettle(face: c.face.id, person: other.id, text: "Moved to \(Fmt.personName(other.name)).") }
                        }
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(uiColor: .systemBackground)))
    }

    private func load() async {
        do {
            page = try await Cloud.api().recognitionConflicts(personId: person.id)
            loadError = nil
        } catch {
            if page == nil { loadError = error }
        }
    }
}

/// Faces the app put on other people that look like the person: move them, or leave them (remembered).
struct PeopleRecognitionElsewhere: View {
    let person: API.PersonOut

    @State private var items: [API.RecognitionElsewhere]?
    @State private var loadError: Error?

    private var name: String { PeopleText.name(person) }

    var body: some View {
        Group {
            if let items {
                if items.isEmpty {
                    Text("Nothing left here.").font(.caption).foregroundStyle(.secondary)
                } else {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(items, id: \.face.id) { e in row(e) }
                    }
                }
            } else if let loadError {
                Text(ErrorText.describe(loadError, doing: "load them")).font(.caption).foregroundStyle(.red)
            } else {
                Text("Loading.").font(.caption).foregroundStyle(.secondary)
            }
        }
        .task { await load() }
        .onCloudChange { Task { await load() } }
    }

    private func row(_ e: API.RecognitionElsewhere) -> some View {
        let other = Fmt.personName(e.person.name)
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                PeopleRecognitionCrop(face: e.face, label: "Open the photo")
                Text("on \(other), \(Int((e.p * 100).rounded()))% sure it is \(name)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
                Button("This is \(name)") {
                    Task { await peopleSettle(face: e.face.id, person: person.id, text: "Moved to \(name).") }
                }
                .buttonStyle(.bordered)
                Button("Leave") { Task { await leave(e, on: other) } }
                    .buttonStyle(.borderless)
            }
            .controlSize(.small)
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(uiColor: .systemBackground)))
    }

    private func load() async {
        do {
            items = try await Cloud.api().recognitionElsewhere(personId: person.id).items
            loadError = nil
        } catch {
            if items == nil { loadError = error }
        }
    }

    private func leave(_ e: API.RecognitionElsewhere, on other: String) async {
        let name = self.name
        await PeopleAction.run("Left a face where it is", doing: "save the answer") {
            try await Cloud.api().leaveRecognitionElsewhere(personId: person.id, body: API.RecognitionLeaveIn(faceIds: [e.face.id]))
        } message: { _ in
            "Left on \(other). It is not asked about for \(name) again."
        }
    }
}
