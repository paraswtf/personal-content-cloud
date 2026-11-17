import SwiftUI

/// Settings > Learning (the website's LearningTab, docs/ARCHITECTURE.md section 18): what the small models learned
/// from the owner's corrections, from GET /api/system/learning. The switch turns learning on and off; Train now fits
/// the models at once; Apply queues a fresh look at items and documents that were decided automatically; Reset
/// deletes the learned models (never the choices themselves). The Vault card holds the vault sensitivity. While
/// tattoos and nails are read, or photos taken together are lined up, the status is asked again every 15 seconds.
struct SettingsLearningView: View {
    @State private var status: API.LearningStatus?
    @State private var forwards: SettingsSystemStatus.Forwards?
    @State private var loadError: Error?
    @State private var switching = false
    @State private var savingSensitivity = false
    @State private var training = false
    @State private var applying = false
    @State private var resetting = false
    @State private var confirmApply = false
    @State private var confirmReset = false

    var body: some View {
        Group {
            if let status {
                content(status, SettingsLearners(status.learners))
            } else if let loadError {
                ErrorState(error: loadError) { Task { await load() } }
            } else {
                LoadingState(text: "Loading learning status")
            }
        }
        .navigationTitle("Learning")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await load()
            await loadForwards()
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 15_000_000_000)
                if let status, SettingsLearners(status.learners).scanning { await load() }
            }
        }
        .confirmAction("Apply to existing items", isPresented: $confirmApply, actionTitle: "Apply", role: nil,
                       message: "Items whose category, vault or mis-click decision was made automatically are analysed again, documents typed automatically are read again, and learned tags are suggested on look-alikes, so they follow what was learned. Choices you made yourself stay as they are. On a large library this takes a while.") {
            Task { await apply() }
        }
        .confirmAction("Reset what was learned", isPresented: $confirmReset, actionTitle: "Reset",
                       message: "The learned models are deleted and automatic decisions go back to the built-in rules. Your corrections are kept, so the next training learns them again. To stop learning, turn the switch off.") {
            Task { await reset() }
        }
    }

    private func content(_ s: API.LearningStatus, _ l: SettingsLearners) -> some View {
        List {
            switchSection(s)
            learnedSection(s, l)
            correctionsSection(s)
            useSection(s, l)
            Section {
                Text("Make corrections where you always have: change a photo's category or move it to the vault on the item, keep or delete flagged photos in the mis-click review, keep or mark forwards in the forwards review, fix a document's type or fields on the document, and assign faces, confirm or reject guesses on the people pages. Add your own tags on an item, and accept or remove the tags it suggests. Your own choices are never changed by what is learned.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } header: {
                SettingsSectionHeader(title: "How to teach it") { EmptyView() }
            }
        }
        .accessibilityIdentifier("learning.list")
        .refreshable { await load() }
    }

    // MARK: Sections

    private func switchSection(_ s: API.LearningStatus) -> some View {
        Section {
            Toggle("Learn from my choices", isOn: Binding(get: { s.enabled }, set: { on in Task { await setEnabled(on) } }))
                .disabled(switching)
                .accessibilityIdentifier("learning.enabled")
            Text(SettingsSystemText.trainedLine(s))
                .font(.subheadline)
                .foregroundStyle(.secondary)
            if !s.enabled {
                Text("Learning is off. The built-in rules decide on their own, and corrections are still recorded for later.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        } footer: {
            Text("Every correction you make teaches small models that run on this server, so the next automatic decision of the same kind follows your choice. Nothing is sent anywhere.")
        }
    }

    @ViewBuilder
    private func learnedSection(_ s: API.LearningStatus, _ l: SettingsLearners) -> some View {
        Section {
            if l.nothingLearned {
                VStack(spacing: 8) {
                    Image(systemName: "graduationcap")
                        .font(.title)
                        .foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                    Text("Nothing learned yet")
                        .font(.headline)
                    Text("Correct a category, a document type, a mis-click, a forward or a person, or add your own tags, and the next training learns from it.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .accessibilityElement(children: .combine)
                vaultCard(s, l)
            } else {
                docTypesCard(l)
                categoriesCard(l)
                vaultCard(s, l)
                misclicksCard(l)
                peopleCard(l)
                tagsCard(l)
                forwardsCard(l)
                if let traits = l.traits { traitsCard(traits) }
                if let place = l.placeScan { placeCard(place) }
            }
        } header: {
            SettingsSectionHeader(title: "What has been learned") { EmptyView() }
        }
    }

    private func correctionsSection(_ s: API.LearningStatus) -> some View {
        let total = SettingsSystemText.eventKinds.reduce(0) { $0 + (s.events[$1.key] ?? 0) }
        return Section {
            if total == 0 {
                Text("None yet.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(SettingsSystemText.eventKinds, id: \.key) { kind in
                    LabeledContent {
                        Text((s.events[kind.key] ?? 0).formatted())
                            .monospacedDigit()
                    } label: {
                        Text(kind.label)
                    }
                }
            }
        } header: {
            SettingsSectionHeader(title: "Corrections recorded") { EmptyView() }
        } footer: {
            Text("Corrections and guess outcomes recorded since learning was added. Choices saved earlier on items, documents and people are used for training as well.")
        }
    }

    private func useSection(_ s: API.LearningStatus, _ l: SettingsLearners) -> some View {
        Section {
            SettingsActionRow(title: "Train now", systemImage: "text.magnifyingglass", busy: training) {
                Task { await train() }
            }
            .disabled(!s.enabled)
            .accessibilityIdentifier("learning.train")
            SettingsActionRow(title: "Apply to existing items", systemImage: "wand.and.stars", busy: applying) {
                confirmApply = true
            }
            .disabled(!s.enabled || l.nothingLearned)
            .accessibilityIdentifier("learning.apply")
            SettingsActionRow(title: "Reset", systemImage: "arrow.counterclockwise", busy: resetting, role: .destructive) {
                confirmReset = true
            }
            .accessibilityIdentifier("learning.reset")
        } header: {
            SettingsSectionHeader(title: "Use it now") { EmptyView() }
        } footer: {
            Text("Training also runs on its own every 10 minutes when there are new corrections. Learned models only shape new decisions; apply them to reach items already in the library.")
        }
    }

    // MARK: Learner cards

    private func docTypesCard(_ l: SettingsLearners) -> some View {
        SettingsLearnerCard(systemImage: "doc.text", title: "Document types", learned: l.docTypeExamples > 0) {
            Text(SettingsLearningText.docTypes(l))
        }
    }

    private func categoriesCard(_ l: SettingsLearners) -> some View {
        SettingsLearnerCard(systemImage: "square.on.circle", title: "Categories", learned: l.categoryExamples > 0) {
            Text(SettingsLearningText.categories(l))
        }
    }

    private func vaultCard(_ s: API.LearningStatus, _ l: SettingsLearners) -> some View {
        SettingsLearnerCard(systemImage: "eye.slash", title: "Vault", learned: l.vaultExamples > 0 || l.vaultWeights) {
            VStack(alignment: .leading, spacing: 8) {
                Text(vaultText(l))
                Picker("Vault sensitivity", selection: Binding(get: { s.vaultSensitivity },
                                                                set: { v in
                                                                    if v != s.vaultSensitivity { Task { await setSensitivity(v) } }
                                                                })) {
                    Text("Low").tag(API.VaultSensitivity.low)
                    Text("Normal").tag(API.VaultSensitivity.normal)
                    Text("High").tag(API.VaultSensitivity.high)
                }
                .pickerStyle(.segmented)
                .disabled(savingSensitivity)
                .accessibilityIdentifier("learning.sensitivity")
                Text(SettingsSystemText.sensitivityHint(s.vaultSensitivity))
            }
        }
    }

    private func vaultText(_ l: SettingsLearners) -> String {
        var text: String
        if l.vaultExamples > 0 {
            if let threshold = l.vaultThreshold {
                text = "\(Fmt.count(l.vaultExamples, "example")), threshold \(String(format: "%.2f", threshold))."
            } else {
                text = "\(Fmt.count(l.vaultExamples, "example")). Close look-alikes follow your choice. A threshold of its own needs at least 8 examples, with 3 or more on each side."
            }
        } else {
            text = "Nothing yet. Move a photo into or out of the vault to teach it."
        }
        if l.vaultWeights { text += " Weights learned from your corrections." }
        return text
    }

    private func misclicksCard(_ l: SettingsLearners) -> some View {
        let learned = !SettingsSystemText.misclickChanges(l.misclickThresholds).isEmpty
        return SettingsLearnerCard(systemImage: "photo.badge.exclamationmark", title: "Mis-clicks", learned: learned) {
            Text(SettingsLearningText.misclicks(l))
        }
    }

    private func peopleCard(_ l: SettingsLearners) -> some View {
        SettingsLearnerCard(systemImage: "person.2", title: "People", learned: l.peoplePersons > 0 || l.peopleWeights) {
            VStack(alignment: .leading, spacing: 4) {
                Text(l.peoplePersons > 0
                     ? "\(Fmt.count(l.peoplePersons, "person", "people")) with their own match threshold."
                     : "No person has their own match threshold yet. Assign faces to people to teach it.")
                Text(peopleWeightsText(l))
            }
        }
    }

    private func peopleWeightsText(_ l: SettingsLearners) -> String {
        if l.peopleWeights {
            return "Guess weights learned from \(Fmt.count(l.peopleEvents, "confirmation and rejection", "confirmations and rejections"))."
        }
        if l.peopleEvents >= 30 {
            return "Guess weights: not yet. \(Fmt.count(l.peopleEvents, "guess", "guesses")) decided; both confirmed and rejected guesses are needed."
        }
        return "Guess weights: not yet, \(l.peopleEvents) of 30 confirmed or rejected guesses needed."
    }

    private func tagsCard(_ l: SettingsLearners) -> some View {
        SettingsLearnerCard(systemImage: "tag", title: "Tags", learned: (l.tags?.tags ?? 0) > 0) {
            Text(SettingsLearningText.tags(l))
        }
    }

    private func forwardsCard(_ l: SettingsLearners) -> some View {
        let n = l.forwards?.examples ?? 0
        let learned = l.forwards?.learned ?? false
        return SettingsLearnerCard(systemImage: "arrowshape.turn.up.right", title: "Forwards", learned: learned) {
            VStack(alignment: .leading, spacing: 4) {
                Text(SettingsLearningText.forwards(examples: n, learned: learned))
                if let backlog = forwards, backlog.unchecked > 0 || backlog.running {
                    Text(SettingsLearningText.forwardsBacklog(backlog))
                }
            }
        }
    }

    private func traitsCard(_ t: SettingsLearners.Traits) -> some View {
        SettingsLearnerCard(systemImage: "pencil.and.scribble", title: "Tattoos and nails", learned: t.adjusted) {
            VStack(alignment: .leading, spacing: 4) {
                Text(traitsText(t))
                if let scan = t.scan {
                    if !scan.enabled {
                        Text("Reading tattoos and nails is turned off.")
                    } else if !scan.paused.isEmpty {
                        Text("Paused: a model is missing (\(scan.paused.joined(separator: ", "))).")
                    } else if t.reading {
                        Text("Reading people boxes: \(scan.scanned.formatted()) of \(scan.eligible.formatted()). Tattoos and nails count in guesses once all are read.")
                        if let error = scan.lastError {
                            Text("Stopped on an error: \(error). It tries again later.")
                                .foregroundStyle(.red)
                        }
                    }
                }
            }
        }
    }

    private func traitsText(_ t: SettingsLearners.Traits) -> String {
        var text: String
        if t.adjusted {
            let pairs = t.pairs ?? t.hardNegatives ?? 0
            if let outcomes = t.outcomes, pairs > 0 {
                text = "Tattoos and nails: \(Fmt.count(outcomes, "outcome")), thresholds from \(Fmt.count(pairs, "pair"))."
            } else if let outcomes = t.outcomes {
                text = "Tattoos and nails: \(Fmt.count(outcomes, "outcome")). Thresholds adjusted to your answers."
            } else {
                text = "Thresholds adjusted to your library and your answers."
            }
            text += " A tattoo match needs \(t.tattooMinInliers) matching points."
        } else {
            text = t.reading ? "Using the defaults. Guesses from tattoos alone start once every photo is read." : "Using the defaults."
            if let outcomes = t.outcomes, outcomes > 0 {
                text += " \(Fmt.count(outcomes, "guess", "guesses")) from tattoos alone answered so far."
            }
        }
        return text
    }

    private func placeCard(_ scan: SettingsLearners.Scan) -> some View {
        let off = !scan.enabled
        let lining = !off && !scan.ready && scan.scanned < scan.eligible
        return SettingsLearnerCard(systemImage: "figure.walk", title: "Position", learned: !off && scan.ready,
                                   labels: ("Ready", off ? "Off" : "Working")) {
            VStack(alignment: .leading, spacing: 4) {
                Text(off
                     ? "Following people from one photo to the next is turned off."
                     : "A person who stands in the same spot in photos taken moments apart is followed from one to the next. Nothing is learned here: it counts as a reason for a guess, with clothes.")
                if lining {
                    Text("Lining up photos taken together: \(scan.scanned.formatted()) of \(scan.eligible.formatted()).")
                }
                if !off, let error = scan.lastError {
                    Text("Stopped on an error: \(error). It tries again later.")
                        .foregroundStyle(.red)
                }
            }
        }
    }

    // MARK: Server calls

    private func load() async {
        do {
            let answer = try await Cloud.api().learningStatus()
            status = answer
            loadError = nil
        } catch {
            if SettingsSystemView.cancelled(error) { return }
            if status == nil { loadError = error }
        }
    }

    /// The forwards card's line about older items comes from GET /api/system.
    private func loadForwards() async {
        guard let system = try? await Cloud.api().system() else { return }
        forwards = SettingsSystemStatus(system).forwards
    }

    private func setEnabled(_ on: Bool) async {
        let previous = status
        status?.enabled = on
        switching = true
        defer { switching = false }
        do {
            _ = try await Cloud.api().learningEnabled(body: API.LearningEnabledIn(enabled: on))
            // read the server's answer directly: the screen already shows the new value
            let next = try await Cloud.api().learningStatus()
            status = next
            if on && !next.enabled {
                Toaster.shared.ok("Learning is turned off on the server (LEARNING_ENABLED).")
            } else {
                Toaster.shared.ok(on ? "Learning is on." : "Learning is off. The built-in rules decide on their own.")
            }
        } catch {
            status = previous
            Toaster.shared.error(error, doing: "change the setting")
        }
    }

    private func setSensitivity(_ value: API.VaultSensitivity) async {
        let previous = status
        status?.vaultSensitivity = value
        savingSensitivity = true
        defer { savingSensitivity = false }
        do {
            var next = try await Cloud.api().learningEnabled(body: API.LearningEnabledIn(vaultSensitivity: value))
            let moved = next.vaultMoved
            next.vaultMoved = nil
            status = next
            if (moved?["moved_in"] ?? 0) > 0 || (moved?["moved_out"] ?? 0) > 0 { CloudEvents.post() }
            Toaster.shared.ok("Vault sensitivity set to \(SettingsSystemText.sensitivityLabel(value)). \(SettingsSystemText.movedText(moved))")
        } catch {
            status = previous
            Toaster.shared.error(error, doing: "change the vault sensitivity")
        }
    }

    private func train() async {
        training = true
        defer { training = false }
        do {
            status = try await Cloud.api().learningTrainNow()
            Toaster.shared.ok("Training finished.")
        } catch {
            Toaster.shared.error(error, doing: "train now")
        }
    }

    private func apply() async {
        applying = true
        defer { applying = false }
        do {
            let out = try await Cloud.api().learningApply()
            Toaster.shared.ok(SettingsSystemText.applyMessage(items: out.items, documents: out.documents, tags: out.tags))
            if out.tags > 0 || out.items + out.documents > 0 { CloudEvents.post() }
        } catch {
            Toaster.shared.error(error, doing: "queue the items")
        }
    }

    private func reset() async {
        resetting = true
        defer { resetting = false }
        do {
            _ = try await Cloud.api().learningReset()
            Toaster.shared.ok("What was learned has been deleted. Your corrections are kept.")
            await load()
        } catch {
            Toaster.shared.error(error, doing: "reset")
        }
    }
}

/// One learner: what it learned, with a badge saying whether it learned anything yet.
struct SettingsLearnerCard<Content: View>: View {
    let systemImage: String
    let title: String
    let learned: Bool
    var labels: (learned: String, notYet: String) = ("Learned", "Not yet")
    @ViewBuilder let content: () -> Content

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: systemImage)
                .foregroundStyle(.secondary)
                .frame(width: 24)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .firstTextBaseline) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                    Spacer(minLength: 8)
                    SettingsBadge(text: learned ? labels.learned : labels.notYet, tone: learned ? .ok : .idle)
                }
                content()
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

/// The learner cards' sentences (the website's LearningTab), kept apart from the views.
enum SettingsLearningText {
    static func docTypes(_ l: SettingsLearners) -> String {
        guard l.docTypeExamples > 0 else {
            return "Nothing yet. Change the type of a document, or edit its fields, to teach it."
        }
        let names: (String) -> String = { SettingsSystemText.docTypeLabel($0) }
        let list = l.docTypeLabels.isEmpty ? "" : ": " + SettingsSystemText.labelList(l.docTypeLabels, names)
        return "Learned from \(Fmt.count(l.docTypeExamples, "correction")), \(Fmt.count(l.docTypeLabels.count, "type"))\(list)."
    }

    static func categories(_ l: SettingsLearners) -> String {
        guard l.categoryExamples > 0 else { return "Nothing yet. Change the category of a photo to teach it." }
        let names: (String) -> String = { SettingsSystemText.categoryLabel($0) }
        let across = l.categoryLabels.isEmpty ? "" : " across " + SettingsSystemText.labelList(l.categoryLabels, names)
        return "\(Fmt.count(l.categoryExamples, "example"))\(across)."
    }

    static func misclicks(_ l: SettingsLearners) -> String {
        let changes = SettingsSystemText.misclickChanges(l.misclickThresholds)
        if !changes.isEmpty {
            let joined = changes.joined(separator: "; ")
            let sentence = joined.prefix(1).uppercased() + String(joined.dropFirst())
            return "\(sentence), from \(Fmt.count(l.misclickExamples, "decision"))."
        }
        if l.misclickExamples > 0 {
            return "\(Fmt.count(l.misclickExamples, "decision")) so far. The built-in thresholds still fit them; a change needs at least 5 kept and 5 confirmed photos of the same kind."
        }
        return "Nothing yet. Keep a flagged photo, or delete it, in the mis-click review to teach it."
    }

    static func tags(_ l: SettingsLearners) -> String {
        guard let t = l.tags else { return "Nothing yet. Add your own tags to items, and look-alikes get them suggested." }
        if t.tags > 0 {
            return "\(Fmt.count(t.tags, "tag")) learned from \(Fmt.count(t.examples, "tagged item")), \(Fmt.count(t.rejections, "removed suggestion"))."
        }
        if t.examples > 0 {
            let removed = t.rejections > 0 ? ", " + Fmt.count(t.rejections, "removed suggestion") : ""
            return "\(Fmt.count(t.examples, "tagged item")) so far\(removed). A tag is learned once 3 photos and videos, or 3 documents, carry it."
        }
        return "Nothing yet. Add your own tags to items, and look-alikes get them suggested."
    }

    static func forwards(examples n: Int, learned: Bool) -> String {
        if learned {
            return "Learned from \(Fmt.count(n, "decision")): what comes through a messaging app is set apart or kept the way you decided."
        }
        if n > 0 {
            return "\(Fmt.count(n, "decision")) so far. The built-in rule decides until 10 decisions with both answers exist."
        }
        return "Nothing yet. Keep a forward, or mark one, in the forwards review to teach it."
    }

    static func forwardsBacklog(_ b: SettingsSystemStatus.Forwards) -> String {
        let head = b.running ? "Checking older items for forwards" : "Older items waiting for a check"
        return b.unchecked > 0 ? "\(head): \(b.unchecked.formatted()) to go." : "\(head)."
    }
}
