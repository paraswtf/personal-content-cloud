import SwiftUI

/// A person's "Tattoos and nails" tab (the website's TraitsTab, docs/ARCHITECTURE.md section 34): their tattoos
/// (long term, each with the photos it was matched in and when it was first and last seen), and the mehndi and nails
/// of the last 60 days (short term). Each card has "Not a tattoo" (nails, mehndi) and "Not NAME's" for its crop;
/// a crop opens its photo, a tattoo's photos open as one run in the viewer.
struct PeopleTraitsTab: View {
    let person: API.PersonOut
    let header: AnyView

    @Environment(AppRouter.self) private var router
    @State private var traits: API.PersonTraitsOut?
    @State private var loadError: Error?
    @State private var busy: Int?

    private var owner: String? { Fmt.isNamed(person.name) ? person.name : nil }
    private var who: String { owner ?? "this person" }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                content.padding(.horizontal, 16)
            }
            .padding(.bottom, 16)
        }
        .task { await load() }
        .refreshable { await load() }
        .onCloudChange { Task { await load() } }
    }

    @ViewBuilder private var content: some View {
        if let t = traits {
            if t.tattoos.isEmpty && t.mehndi.isEmpty && t.nails.isEmpty {
                EmptyState(title: "No tattoos or nails yet", systemImage: "pencil.tip",
                           text: "Tattoos, nails and mehndi matched to \(who) show here once their photos are read.")
            } else {
                if !t.tattoos.isEmpty {
                    section("Tattoos", systemImage: "pencil.tip",
                            note: "Long term: the same tattoo helps find \(who) in photos taken months or years apart.") {
                        ForEach(t.tattoos, id: \.best.id) { g in
                            card(g.best, title: g.label, sub: seenLine(g),
                                 uids: g.itemUids.isEmpty ? [g.best.itemUid].compactMap { $0 } : g.itemUids,
                                 first: g.best.itemUid ?? g.itemUids.first)
                        }
                    }
                }
                if !t.mehndi.isEmpty {
                    section("Mehndi lately", systemImage: "hand.raised",
                            note: "Short term: mehndi helps only for photos taken within about three weeks.") {
                        ForEach(t.mehndi, id: \.id) { shortCard($0) }
                    }
                }
                if !t.nails.isEmpty {
                    section("Nails lately", systemImage: "hand.raised",
                            note: "Short term: nail polish only adds weight to other signs, for photos taken within about two weeks.") {
                        ForEach(t.nails, id: \.id) { shortCard($0) }
                    }
                }
            }
        } else if let loadError {
            ErrorState(error: loadError) { Task { await load() } }
        } else {
            LoadingState()
        }
    }

    private func section<Content: View>(_ title: String, systemImage: String, note: String,
                                        @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            PeopleSectionHeader(title: title, systemImage: systemImage, note: note)
            content()
        }
    }

    private func shortCard(_ t: API.TraitOut) -> some View {
        card(t, title: t.label, sub: t.takenAt != nil ? Fmt.date(t.takenAt) : "Date not known",
             uids: [t.itemUid].compactMap { $0 }, first: t.itemUid)
    }

    private func card(_ t: API.TraitOut, title: String, sub: String, uids: [String], first: String?) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Button {
                if let first { PeopleViewer.open(router, uids: uids, start: first) }
            } label: {
                RemoteImage(ref: t.cropUrl, maxPixel: 240)
                    .frame(width: 80, height: 80)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .opacity(t.hidden ? 0.5 : 1)
            }
            .buttonStyle(.plain)
            .disabled(first == nil)
            .accessibilityLabel(first == nil ? t.label : "Open the photo: \(t.label)")
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    if t.kind == .nails, let color = PeopleTraitColor.color(t.hex) {
                        Circle().fill(color).frame(width: 14, height: 14)
                            .overlay(Circle().stroke(Color(uiColor: .separator)))
                            .accessibilityHidden(true)
                    }
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                        .strikethrough(t.hidden)
                        .foregroundStyle(t.hidden ? Color.secondary : Color.primary)
                }
                Text(sub).font(.caption).foregroundStyle(.secondary)
                if t.hidden {
                    Text("Hidden: it does not count for guesses.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
            Menu {
                if t.hidden {
                    Button {
                        Task { await hide(t, verdict: nil) }
                    } label: {
                        Label("Show again", systemImage: "eye")
                    }
                } else {
                    Button {
                        Task { await hide(t, verdict: .notTrait) }
                    } label: {
                        Label(notTraitLabel(t.kind), systemImage: "eye.slash")
                    }
                    Button {
                        Task { await hide(t, verdict: .notTheirs) }
                    } label: {
                        Label(owner.map { "Not \($0)'s" } ?? "Not this person's", systemImage: "person.crop.circle.badge.xmark")
                    }
                }
            } label: {
                Image(systemName: "ellipsis").frame(width: 40, height: 40)
            }
            .disabled(busy == t.id)
            .accessibilityLabel("Options for \(t.label)")
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(uiColor: .secondarySystemBackground)))
    }

    /// "In 14 photos, March 2025 to October 2026".
    private func seenLine(_ g: API.TraitGroupOut) -> String {
        let from = monthYear(g.firstSeen), to = monthYear(g.lastSeen)
        let span: String
        if !from.isEmpty && !to.isEmpty && from != to {
            span = ", \(from) to \(to)"
        } else if !from.isEmpty || !to.isEmpty {
            span = ", \(from.isEmpty ? to : from)"
        } else {
            span = ""
        }
        return "In \(Fmt.count(g.photos, "photo"))\(span)"
    }

    private func monthYear(_ iso: String?) -> String {
        guard let d = ISODate.parse(iso) else { return "" }
        return d.formatted(.dateTime.month(.wide).year())
    }

    private func word(_ kind: API.TraitGroupOutKind) -> String {
        switch kind {
        case .tattoo: return "tattoo"
        case .mehndi: return "mehndi"
        default: return "nail polish"
        }
    }

    private func notTraitLabel(_ kind: API.TraitGroupOutKind) -> String {
        switch kind {
        case .tattoo: return "Not a tattoo"
        case .mehndi: return "Not mehndi"
        default: return "Not nail polish"
        }
    }

    private func load() async {
        do {
            traits = try await Cloud.api().personTraits(personId: person.id)
            loadError = nil
        } catch {
            if traits == nil { loadError = error }
        }
    }

    private func hide(_ t: API.TraitOut, verdict: API.Verdict?) async {
        busy = t.id
        defer { busy = nil }
        let owner = self.owner
        let word = self.word(t.kind)
        let label: String
        switch verdict {
        case .none: label = "Shown again on this photo: \(t.label)."
        case .some(.notTrait): label = "Not counted as \(t.kind == .tattoo ? "a tattoo" : word) any more: \(t.label)."
        default: label = "Not counted as \(owner.map { "\($0)'s" } ?? "this person's") \(word) any more: \(t.label)."
        }
        let request = verdict.map { API.TraitHideIn(hidden: true, verdict: $0) } ?? API.TraitHideIn(hidden: false)
        let r = await PeopleAction.run(label, doing: "save that", items: [t.itemUid].compactMap { $0 }) {
            try await Cloud.api().hideTrait(traitId: t.id, body: request)
        } message: { _ in
            label
        }
        if r != nil { await load() }
    }
}

/// A nail polish swatch from the server's "#rrggbb" (anything else shows no swatch).
enum PeopleTraitColor {
    static func color(_ hex: String?) -> Color? {
        guard var text = hex?.trimmingCharacters(in: .whitespaces), text.hasPrefix("#") else { return nil }
        text.removeFirst()
        guard text.count == 6, let value = UInt32(text, radix: 16) else { return nil }
        return Color(red: Double((value >> 16) & 0xFF) / 255, green: Double((value >> 8) & 0xFF) / 255,
                     blue: Double(value & 0xFF) / 255)
    }
}
