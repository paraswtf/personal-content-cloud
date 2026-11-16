import SwiftUI

/// A person or pet tile (the website's PersonCard): round face or animal crop, name, item count and a badge for open
/// guesses. With `selected` set, it shows a check for picking people to merge.
struct PeopleCard: View {
    let person: API.PersonOut
    /// nil: not picking; true or false: picked or not.
    var selected: Bool?

    var body: some View {
        VStack(spacing: 6) {
            ZStack(alignment: .topTrailing) {
                Avatar(name: person.name, ref: person.coverUrl, size: 76)
                    .overlay(Circle().stroke(Color.accentColor, lineWidth: selected == true ? 3 : 0))
                    .overlay(alignment: .bottom) {
                        if person.guessCount > 0 {
                            Text("\(person.guessCount) \(person.guessCount == 1 ? "guess" : "guesses")")
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Capsule().fill(Color.orange))
                                .offset(y: 6)
                        }
                    }
                if let selected {
                    Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                        .font(.title3)
                        .foregroundStyle(selected ? Color.accentColor : Color.secondary)
                        .background(Circle().fill(Color(uiColor: .systemBackground)))
                }
            }
            .padding(.bottom, person.guessCount > 0 ? 4 : 0)
            Text(PeopleText.name(person))
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Fmt.isNamed(person.name) ? Color.primary : Color.secondary)
                .lineLimit(1)
            HStack(spacing: 3) {
                if PeopleText.isPet(person) { Image(systemName: "pawprint.fill") }
                Text(detail)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .padding(.horizontal, 4)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(selected == true ? Color.accentColor.opacity(0.14) : Color(uiColor: .secondarySystemBackground))
        )
        .contentShape(RoundedRectangle(cornerRadius: 16))
    }

    private var detail: String {
        PeopleText.isPet(person)
            ? "\(Fmt.speciesLabel(person.species)), \(Fmt.count(person.itemCount, "item"))"
            : Fmt.count(person.itemCount, "item")
    }

    /// What VoiceOver reads, starting with the name (the screen tests find a card by it).
    static func label(_ person: API.PersonOut) -> String {
        var parts = [PeopleText.name(person)]
        if PeopleText.isPet(person) { parts.append(Fmt.speciesLabel(person.species)) }
        parts.append(Fmt.count(person.itemCount, "item"))
        if person.guessCount > 0 { parts.append("\(Fmt.count(person.guessCount, "guess", "guesses")) to review") }
        return parts.joined(separator: ", ")
    }
}

/// A grid of people or pets: links to their pages, or, while picking, buttons that pick them.
struct PeopleGrid: View {
    let rows: [PeopleRow]
    var picking: SelectionModel<Int>?
    var onAppear: ((PeopleRow) -> Void)?

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 104), spacing: 10)], spacing: 10) {
            ForEach(rows) { row in
                Group {
                    if let picking, picking.active {
                        Button {
                            picking.toggle(row.id)
                        } label: {
                            PeopleCard(person: row.person, selected: picking.contains(row.id))
                        }
                        .buttonStyle(.plain)
                        .accessibilityAddTraits(picking.contains(row.id) ? .isSelected : [])
                    } else {
                        NavigationLink(value: Route.person(id: row.id)) {
                            PeopleCard(person: row.person)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .accessibilityLabel(PeopleCard.label(row.person))
                .accessibilityIdentifier("people.card")
                .onAppear { onAppear?(row) }
            }
        }
    }
}
