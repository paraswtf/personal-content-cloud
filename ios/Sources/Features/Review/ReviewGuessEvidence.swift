import SwiftUI

/// Why a guess names who it names (the website's evidence.ts, TraitCompare.tsx and PlaceCompare.tsx): the line
/// of a clothes, face or grouped guess, the strongest trait reason with this photo's crop and the guessed person's
/// side by side, the position reason with the two boxes, and the other signs folded under "N more signs".
/// The review area's guess card shows it, and the boxes area's guess chip and face editor. `cropUrl` is the crop
/// of the box the guess is for (the position comparison shows it); `dark` draws it for the viewer's black
/// background; `onOpenItem` opens the photo a reference crop comes from (without it the crops are plain pictures).
struct ReviewGuessEvidence: View {
    let guess: API.GuessOut
    var cropUrl: String? = nil
    var dark = false
    var vault = false
    var onOpenItem: ((String) -> Void)? = nil

    var body: some View {
        let name = ReviewEvidence.displayName(guess.person)
        let reasons = ReviewEvidence.traitReasons(guess)
        let line = ReviewEvidence.clothesLine(guess)
        let place = ReviewEvidence.placeReason(guess)
        let more = reasons.dropFirst().map(\.text) + ReviewEvidence.againstReasons(guess)
        let moreTitle: String = more.count == 1 ? "1 more sign" : "\(more.count) more signs"
        VStack(alignment: .leading, spacing: 10) {
            if !line.isEmpty {
                Text(line)
                    .font(.caption)
                    .foregroundStyle(secondary)
            }
            if let lead = reasons.first {
                VStack(alignment: .leading, spacing: 6) {
                    Text(lead.text)
                        .font(.caption)
                        .foregroundStyle(secondary)
                    ReviewTraitCompare(reason: lead, name: name, dark: dark, vault: vault, onOpenRef: onOpenItem)
                }
            }
            if let place {
                VStack(alignment: .leading, spacing: 6) {
                    Text(place.text)
                        .font(.caption)
                        .foregroundStyle(secondary)
                    ReviewPlaceCompare(reason: place, cropUrl: cropUrl, name: name, dark: dark, vault: vault,
                                       onOpenRef: onOpenItem)
                }
            }
            if !more.isEmpty {
                DisclosureGroup(moreTitle) {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(more.enumerated()), id: \.offset) { _, text in
                            HStack(alignment: .firstTextBaseline, spacing: 6) {
                                Text("\u{2022}").accessibilityHidden(true)
                                Text(text)
                            }
                            .foregroundStyle(secondary)
                        }
                    }
                    .padding(.top, 4)
                }
                .font(.caption)
                .tint(dark ? .white : .accentColor)
                .accessibilityIdentifier("review.moreSigns")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var secondary: Color { dark ? Color.white.opacity(0.7) : Color.secondary }
}

/// One crop of a comparison, or "Photo changed" when the server no longer has it or it no longer loads (the row
/// was measured again, or the photo went to the vault or the trash), as the website's Crop.
struct ReviewCrop: View {
    let ref: String?
    let label: String
    var dark = false
    var vault = false

    /// The address that failed to load (a new address is tried again).
    @State private var failedRef: String?

    private var changed: Bool { ref == nil || failedRef == ref }

    var body: some View {
        ZStack {
            if let ref, !changed {
                RemoteImage(ref: ref, maxPixel: 216, vault: vault, onFailure: { failedRef = ref })
            } else {
                Text("Photo changed")
                    .font(.system(size: 10))
                    .multilineTextAlignment(.center)
                    .foregroundStyle(dark ? Color.white.opacity(0.6) : Color.secondary)
                    .padding(4)
            }
        }
        .frame(width: 64, height: 64)
        .background(dark ? Color.white.opacity(0.1) : Color(uiColor: .secondarySystemFill))
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(dark ? Color.white.opacity(0.2) : Color(uiColor: .separator), lineWidth: 0.5)
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(changed ? "Photo changed" : label)
    }
}

/// A crop with its caption under it.
struct ReviewCropFigure<Crop: View>: View {
    let caption: String
    var dark = false
    @ViewBuilder let crop: () -> Crop

    var body: some View {
        VStack(spacing: 4) {
            crop()
            Text(caption)
                .font(.system(size: 11))
                .foregroundStyle(dark ? Color.white.opacity(0.6) : Color.secondary)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 96)
        }
    }
}

/// Two trait crops side by side (section 34): this photo's region and the guessed person's that it matched; nails
/// add a swatch of the polish.
struct ReviewTraitCompare: View {
    let reason: ReviewEvidence.TraitReason
    let name: String
    var dark = false
    var vault = false
    var onOpenRef: ((String) -> Void)?

    var body: some View {
        let what = "\(ReviewEvidence.traitNoun(reason.kind)) on the \(ReviewEvidence.partWords(reason.part))"
        let when = reason.refTakenAt.map { Fmt.date($0) } ?? ""
        let refCaption = when.isEmpty ? name : "\(name), \(when)"
        HStack(alignment: .top, spacing: 8) {
            ReviewCropFigure(caption: "This photo", dark: dark) {
                ReviewCrop(ref: reason.cropUrl, label: "\(what), this photo", dark: dark, vault: vault)
            }
            ReviewCropFigure(caption: refCaption, dark: dark) {
                if let onOpenRef, let uid = reason.refItemUid {
                    Button {
                        onOpenRef(uid)
                    } label: {
                        ReviewCrop(ref: reason.refCropUrl, label: "\(what), \(name)'s photo", dark: dark, vault: vault)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Open \(name)'s photo\(when.isEmpty ? "" : " from \(when)")")
                } else {
                    ReviewCrop(ref: reason.refCropUrl, label: "\(what), \(name)'s photo", dark: dark, vault: vault)
                }
            }
            if reason.kind == .nails, let hex = reason.hex, let color = Self.color(hex: hex) {
                Circle()
                    .fill(color)
                    .frame(width: 20, height: 20)
                    .overlay(Circle().stroke(dark ? Color.white.opacity(0.3) : Color(uiColor: .separator), lineWidth: 1))
                    .padding(.top, 4)
                    .accessibilityHidden(true)
            }
        }
    }

    /// "#a1b2c3" as a colour (the evidence keeps only that form).
    static func color(hex: String) -> Color? {
        guard hex.count == 7, hex.hasPrefix("#"), let value = UInt32(hex.dropFirst(), radix: 16) else { return nil }
        return Color(red: Double((value >> 16) & 0xFF) / 255, green: Double((value >> 8) & 0xFF) / 255,
                     blue: Double(value & 0xFF) / 255)
    }
}

/// Two boxes side by side for a position reason (section 35): this photo's box and the box the person was followed
/// from, "This photo" and "Sam, 8 s earlier".
struct ReviewPlaceCompare: View {
    let reason: ReviewEvidence.PlaceReason
    var cropUrl: String?
    let name: String
    var dark = false
    var vault = false
    var onOpenRef: ((String) -> Void)?

    var body: some View {
        let refCaption = reason.when.isEmpty ? name : "\(name), \(reason.when)"
        HStack(alignment: .top, spacing: 8) {
            ReviewCropFigure(caption: "This photo", dark: dark) {
                ReviewCrop(ref: cropUrl, label: "The box in this photo", dark: dark, vault: vault)
            }
            ReviewCropFigure(caption: refCaption, dark: dark) {
                if let onOpenRef, let uid = reason.refItemUid {
                    Button {
                        onOpenRef(uid)
                    } label: {
                        ReviewCrop(ref: reason.refCropUrl, label: "\(name) in the other photo", dark: dark, vault: vault)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Open the photo \(name) was followed from")
                } else {
                    ReviewCrop(ref: reason.refCropUrl, label: "\(name) in the other photo", dark: dark, vault: vault)
                }
            }
        }
    }
}
