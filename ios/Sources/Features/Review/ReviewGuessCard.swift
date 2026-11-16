import SwiftUI

/// One guess to confirm, correct or reject (the website's GuessCard): the face or person crop, the photo it is in
/// (for a box in a video, the frame at the box's time with the box outlined), the guessed person with what the guess
/// rests on and the evidence, and Yes, No and Someone else (the guess is learned as wrong and the box goes to the
/// person picked, in one step). The review area builds it; the person page's Guesses tab shows it too. The card
/// sends the answer itself; `onDecided` runs once the owner answered it, so the list can take the card away.
/// `onOpenItem` opens a photo (the guess's own or the one a reason refers to); without it the card opens the photo
/// in the viewer by itself.
struct ReviewGuessCard: View {
    let guess: API.GuessItem
    var onDecided: (() -> Void)? = nil
    var onOpenItem: ((String) -> Void)? = nil

    @Environment(AppRouter.self) private var router
    @State private var busy: ReviewDecisions.Action?
    @State private var picking = false
    @State private var outcome: String?

    private var box: ReviewBox { ReviewBox(guess.box) }
    private var isFace: Bool { guess.kind == .face }
    private var name: String { ReviewEvidence.displayName(guess.guess.person) }
    /// A box in a video: the time of the frame it is on.
    private var frameTime: Double? {
        guard guess.item.kind == .video, let t = box.frameTime, t.isFinite else { return nil }
        return t
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 12) {
                RemoteImage(ref: box.url, maxPixel: 240)
                    .frame(width: 72, height: 72)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Color(uiColor: .separator), lineWidth: 0.5))
                    .accessibilityLabel(isFace ? "Face" : "Person")
                sceneButton
                VStack(alignment: .leading, spacing: 6) {
                    headline
                    NavigationLink(value: Route.person(id: guess.guess.person.id)) {
                        HStack(spacing: 8) {
                            Avatar(name: name, ref: guess.guess.person.coverUrl, size: 28)
                            Text("\(name)?")
                                .font(.subheadline.weight(.semibold))
                                .lineLimit(1)
                            OriginIcon(origin: .guess)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Maybe \(name). Open their page")
                    .accessibilityIdentifier("review.guess.person")
                }
                Spacer(minLength: 0)
            }
            ReviewGuessEvidence(guess: guess.guess, cropUrl: box.url, onOpenItem: { uid in open(uid) })
            if let outcome {
                Label(outcome, systemImage: "checkmark.circle")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("review.guess.outcome")
            } else {
                buttons
            }
        }
        .padding(12)
        .background(Color(uiColor: .secondarySystemGroupedBackground),
                    in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
            .stroke(Color(uiColor: .separator), lineWidth: 0.5))
        .sheet(isPresented: $picking) {
            PersonPickerSheet(title: "Someone else", kind: .person,
                              like: isFace ? .faces([guess.id]) : .bodies([guess.id]),
                              excluding: [guess.guess.person.id]) { pick in
                picking = false
                Task { await decide(.correct, pick: pick) }
            }
        }
    }

    /// "By clothes, 85% sure", with its icons (and the frame's time for a video).
    private var headline: some View {
        let label = ReviewEvidence.label(isFace: isFace, guess: guess.guess)
        let sure = ReviewEvidence.sure(guess.guess)
        let at = frameTime.map { ", at \(Fmt.duration($0))" } ?? ""
        return HStack(spacing: 4) {
            ForEach(ReviewEvidence.icons(isFace: isFace, guess: guess.guess), id: \.self) { icon in
                Image(systemName: icon.systemImage)
                    .imageScale(.small)
                    .accessibilityHidden(true)
            }
            Text("\(label), \(sure)\(at)")
                .lineLimit(2)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    /// The photo the guess is in; a video shows the frame at the box's time with the box outlined.
    private var sceneButton: some View {
        Button {
            open(guess.item.uid)
        } label: {
            sceneThumb
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(Color(uiColor: .separator), lineWidth: 0.5))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(frameTime.map { "Open the video at \(Fmt.duration($0))" }
            ?? "Open the \(guess.item.kind == .video ? "video" : "photo")")
        .accessibilityIdentifier("review.guess.open")
    }

    @ViewBuilder
    private var sceneThumb: some View {
        if let t = frameTime {
            let aspect = ReviewBox.aspect(guess.item)
            let width = min(144, 72 * aspect)
            RemoteImage(ref: ReviewBox.frameRef(guess.item, t: t), maxPixel: 480)
                .frame(width: width, height: 72)
                .overlay(alignment: .topLeading) {
                    Rectangle()
                        .stroke(Color.accentColor, lineWidth: 2)
                        .frame(width: width * box.w, height: 72 * box.h)
                        .offset(x: width * box.x, y: 72 * box.y)
                        .allowsHitTesting(false)
                }
        } else {
            RemoteImage(ref: guess.item.urls.thumb, maxPixel: 240)
                .frame(width: 72, height: 72)
        }
    }

    private var buttons: some View {
        HStack(spacing: 8) {
            Button {
                Task { await decide(.confirm) }
            } label: {
                buttonLabel("Yes", systemImage: "checkmark", action: .confirm)
            }
            .buttonStyle(.borderedProminent)
            .accessibilityHint("Tags them as \(name)")
            .accessibilityIdentifier("review.guess.yes")
            Button {
                Task { await decide(.reject) }
            } label: {
                buttonLabel("No", systemImage: "xmark", action: .reject)
            }
            .buttonStyle(.bordered)
            .accessibilityHint("Says it is not \(name)")
            .accessibilityIdentifier("review.guess.no")
            Button {
                picking = true
            } label: {
                buttonLabel("Someone else", systemImage: "person.fill.questionmark", action: .correct)
            }
            .buttonStyle(.bordered)
            .accessibilityIdentifier("review.guess.other")
        }
        .controlSize(.regular)
        .disabled(busy != nil)
    }

    private func buttonLabel(_ title: String, systemImage: String, action: ReviewDecisions.Action) -> some View {
        HStack(spacing: 4) {
            if busy == action {
                ProgressView().controlSize(.small)
            } else {
                Image(systemName: systemImage).imageScale(.small)
            }
            Text(title).lineLimit(1).minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity)
    }

    private func decide(_ action: ReviewDecisions.Action, pick: PersonPick? = nil) async {
        guard busy == nil else { return }
        busy = action
        defer { busy = nil }
        do {
            switch action {
            case .confirm:
                try await ReviewDecisions.confirm([guess])
                outcome = "Tagged as \(name)"
            case .reject:
                try await ReviewDecisions.reject([guess])
                outcome = "Not \(name)"
            case .correct:
                guard let pick else { return }
                let out = try await ReviewDecisions.correct([guess], to: pick)
                let shown = out.person.map { Fmt.personName($0.name) } ?? Self.name(of: pick)
                outcome = "Tagged as \(shown)"
            }
            CloudEvents.post(items: [guess.item.uid])
            onDecided?()
        } catch {
            Toaster.shared.error(error, doing: "save the answer")
        }
    }

    private func open(_ uid: String) {
        if let onOpenItem {
            onOpenItem(uid)
        } else if uid == guess.item.uid {
            router.show([guess.item], start: guess.item)
        } else {
            router.open(.item(uid: uid))
        }
    }

    static func name(of pick: PersonPick) -> String {
        switch pick {
        case .existing(let option): return option.name
        case .new(let name, _): return name
        }
    }

}

/// The parts of a guess's box the card needs, from either kind of box.
struct ReviewBox {
    let url: String
    let x: Double
    let y: Double
    let w: Double
    let h: Double
    let frameTime: Double?

    init(_ box: API.FaceOutOrBodyOut) {
        switch box {
        case .faceOut(let f):
            url = f.url; x = f.x; y = f.y; w = f.w; h = f.h; frameTime = f.frameTimeS
        case .bodyOut(let b):
            url = b.url; x = b.x; y = b.y; w = b.w; h = b.h; frameTime = b.frameTimeS
        }
    }

    /// Width over height of an item, 16:9 when the server did not say.
    static func aspect(_ item: API.ItemCard) -> CGFloat {
        guard let w = item.width, let h = item.height, w > 0, h > 0 else { return 16.0 / 9.0 }
        return CGFloat(w) / CGFloat(h)
    }

    /// The address of a video's frame at `t` seconds, preview size (the website's frameUrl): whole milliseconds,
    /// and the version key with an "n" for a vault item.
    static func frameRef(_ item: API.ItemCard, t: Double) -> String {
        let ms = t.isFinite && t > 0 ? (t * 1000).rounded(.down) : 0
        let seconds = String(format: "%.3f", ms / 1000)
        let version = item.nsfw ? "\(item.version)n" : "\(item.version)"
        return "/files/frame/\(APIPath.segment(item.uid))?t=\(seconds)&size=preview&v=\(version)"
    }
}
