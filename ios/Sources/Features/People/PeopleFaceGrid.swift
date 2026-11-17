import SwiftUI

/// A grid of face crops (the website's FaceGrid): the size or the guessed name under each, who put the face there,
/// the cover, and a check while picking. Put it inside a ScrollView.
struct PeopleFaceGrid: View {
    let faces: [PeopleFaceRow]
    /// Picking faces: taps pick instead of opening.
    var selection: SelectionModel<Int>?
    var coverId: Int?
    var onOpen: ((API.FaceListItem) -> Void)?
    var onAppear: ((PeopleFaceRow) -> Void)?

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 96), spacing: 6)], spacing: 6) {
            ForEach(faces) { row in
                tile(row.face)
                    .onAppear { onAppear?(row) }
            }
        }
    }

    private var picking: Bool { selection?.active ?? false }

    private func tile(_ f: API.FaceListItem) -> some View {
        let picked = selection?.contains(f.id) ?? false
        let origin = BoxOrigin.of(face: f)
        return Button {
            if picking { selection?.toggle(f.id) } else { onOpen?(f) }
        } label: {
            Color.clear
                .aspectRatio(1, contentMode: .fit)
                .overlay { RemoteImage(ref: f.url, maxPixel: 256).scaleEffect(picked ? 0.92 : 1) }
                .overlay { LinearGradient(colors: [.clear, .black.opacity(0.7)], startPoint: .center, endPoint: .bottom) }
                .overlay(alignment: .bottom) {
                HStack(spacing: 3) {
                    if f.itemKind == .video { Image(systemName: "film") }
                    Text(caption(f)).lineLimit(1)
                    Spacer(minLength: 0)
                    if let origin {
                        OriginIcon(origin: origin, decorative: true)
                    } else if f.assignedBy == .user {
                        Image(systemName: "lock.fill")
                    }
                }
                .font(.caption2.weight(.medium))
                .foregroundStyle(.white)
                .padding(.horizontal, 6)
                .padding(.bottom, 4)
                }
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.accentColor, lineWidth: picked ? 4 : 0))
            .overlay(alignment: .topLeading) {
                if picking {
                    Image(systemName: picked ? "checkmark.circle.fill" : "circle")
                        .font(.title3)
                        .foregroundStyle(picked ? Color.accentColor : Color.white)
                        .shadow(radius: 2)
                        .padding(5)
                }
            }
            .overlay(alignment: .topTrailing) {
                if coverId == f.id {
                    Text("Cover")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(.black.opacity(0.6)))
                        .padding(5)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(picking ? "\(picked ? "Deselect" : "Select") \(label(f, origin))" : "Open the item of \(label(f, origin))")
        .accessibilityAddTraits(picked ? .isSelected : [])
        .accessibilityIdentifier("people.face")
    }

    private func caption(_ f: API.FaceListItem) -> String {
        if let guess = f.guess, Fmt.isNamed(guess.person.name) { return "\(Fmt.personName(guess.person.name))?" }
        if f.itemKind == .video, let t = f.frameTimeS { return Fmt.duration(t) }
        return "\(f.sizePx) px"
    }

    /// "Face 12, 120 px, from a video, maybe Asha" (the website's tile name).
    private func label(_ f: API.FaceListItem, _ origin: BoxOrigin?) -> String {
        var text = "Face \(f.id), \(f.sizePx) px"
        if f.itemKind == .video { text += ", from a video" }
        if let guess = f.guess, Fmt.isNamed(guess.person.name) {
            text += ", maybe \(Fmt.personName(guess.person.name))"
        } else if let origin {
            text += ", \(PeopleText.originPhrase(origin))"
        } else if f.assignedBy == .user {
            text += ", kept unassigned by you"
        }
        return text
    }
}

/// A grid of a pet's crops (the website's AnimalGrid). Put it inside a ScrollView.
struct PeopleCropGrid: View {
    let crops: [PeopleCropRow]
    var selection: SelectionModel<Int>?
    var onOpen: ((API.AnimalListItem) -> Void)?
    var onAppear: ((PeopleCropRow) -> Void)?

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 96), spacing: 6)], spacing: 6) {
            ForEach(crops) { row in
                tile(row.crop)
                    .onAppear { onAppear?(row) }
            }
        }
    }

    private var picking: Bool { selection?.active ?? false }

    private func tile(_ a: API.AnimalListItem) -> some View {
        let picked = selection?.contains(a.id) ?? false
        let origin = BoxOrigin.of(crop: a)
        var label = "\(Fmt.speciesLabel(a.species)) \(a.id)"
        if a.itemKind == .video { label += ", from a video" }
        if let origin { label += ", \(PeopleText.originPhrase(origin))" }
        return Button {
            if picking { selection?.toggle(a.id) } else { onOpen?(a) }
        } label: {
            Color.clear
                .aspectRatio(1, contentMode: .fit)
                .overlay { RemoteImage(ref: a.url, maxPixel: 256).scaleEffect(picked ? 0.92 : 1) }
                .overlay { LinearGradient(colors: [.clear, .black.opacity(0.7)], startPoint: .center, endPoint: .bottom) }
                .overlay(alignment: .bottom) {
                HStack(spacing: 3) {
                    if a.itemKind == .video { Image(systemName: "film") }
                    Text(a.itemKind == .video && a.frameTimeS != nil ? Fmt.duration(a.frameTimeS) : Fmt.speciesLabel(a.species))
                        .lineLimit(1)
                    Spacer(minLength: 0)
                    if let origin { OriginIcon(origin: origin, decorative: true) }
                }
                .font(.caption2.weight(.medium))
                .foregroundStyle(.white)
                .padding(.horizontal, 6)
                .padding(.bottom, 4)
                }
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.accentColor, lineWidth: picked ? 4 : 0))
            .overlay(alignment: .topLeading) {
                if picking {
                    Image(systemName: picked ? "checkmark.circle.fill" : "circle")
                        .font(.title3)
                        .foregroundStyle(picked ? Color.accentColor : Color.white)
                        .shadow(radius: 2)
                        .padding(5)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(picking ? "\(picked ? "Deselect" : "Select") \(label)" : "Open the item of \(label)")
        .accessibilityAddTraits(picked ? .isSelected : [])
        .accessibilityIdentifier("people.crop")
    }
}
