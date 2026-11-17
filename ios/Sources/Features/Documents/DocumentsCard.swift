import SwiftUI

/// One parsed document in the list (the website's DocumentCard): the type's colour and symbol, its title, the
/// person or issuer, the date, a few key fields (identifiers masked) and the ID photo when one was found.
struct DocumentsCard: View {
    let item: API.ItemCard
    let document: API.DocumentOut
    let onOpen: () -> Void

    var body: some View {
        let group = DocumentsMeta.group(document.docGroup)
        let title = DocumentsMeta.title(document)
        Button(action: onOpen) {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 8) {
                    Image(systemName: DocumentsMeta.typeSymbol(document.docType, group: document.docGroup))
                        .imageScale(.small)
                    Text((document.label.isEmpty ? DocumentsMeta.typeLabel(document.docType) : document.label).uppercased())
                        .font(.caption.weight(.semibold))
                        .tracking(0.6)
                        .lineLimit(1)
                    Spacer(minLength: 6)
                    if let date = document.docDate, !date.isEmpty {
                        Text(Fmt.plainDate(date))
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.85))
                    }
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(LinearGradient(colors: [group.from, group.to], startPoint: .leading, endPoint: .trailing))

                HStack(alignment: .top, spacing: 12) {
                    picture
                    VStack(alignment: .leading, spacing: 3) {
                        Text(title)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        if let person = document.personName, !person.isEmpty {
                            Text(person).font(.subheadline).foregroundStyle(.secondary).lineLimit(1)
                        } else if let issuer = document.issuer, !issuer.isEmpty {
                            Text(issuer).font(.subheadline).foregroundStyle(.secondary).lineLimit(1)
                        }
                        VStack(alignment: .leading, spacing: 2) {
                            ForEach(DocumentsMeta.keyFields(document)) { field in
                                HStack(spacing: 8) {
                                    Text(field.label)
                                        .foregroundStyle(.tertiary)
                                        .frame(width: 92, alignment: .leading)
                                        .lineLimit(1)
                                    Text(field.value)
                                        .fontWeight(.medium)
                                        .monospacedDigit()
                                        .foregroundStyle(.primary)
                                        .lineLimit(1)
                                }
                                .font(.caption)
                            }
                        }
                        .padding(.top, 4)
                    }
                    Spacer(minLength: 0)
                }
                .padding(12)
            }
            .background(Color(uiColor: .secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(Color(uiColor: .separator), lineWidth: 0.5))
            .contentShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .ignore)
        .accessibilityAddTraits(.isButton)
        .accessibilityLabel(accessibilityText(title))
        .accessibilityValue(DocumentsMeta.keyFields(document).map { "\($0.label) \($0.value)" }.joined(separator: ", "))
        .accessibilityIdentifier("documents.card")
    }

    /// The ID photo when the parser found one, else the item's thumb (a page of the document).
    @ViewBuilder
    private var picture: some View {
        if let photo = document.photoUrl, !photo.isEmpty {
            RemoteImage(ref: photo, maxPixel: 240)
                .frame(width: 64, height: 80)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color(uiColor: .separator), lineWidth: 0.5))
        } else {
            RemoteImage(ref: item.thumbRef, maxPixel: 240, contentMode: .fit)
                .frame(width: 64, height: 80, alignment: .top)
                .background(Color.white)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color(uiColor: .separator), lineWidth: 0.5))
        }
    }

    /// "Open PAN card, the holder's name" (the website's label).
    private func accessibilityText(_ title: String) -> String {
        if let person = document.personName, !person.isEmpty { return "Open \(title), \(person)" }
        return "Open \(title)"
    }
}
