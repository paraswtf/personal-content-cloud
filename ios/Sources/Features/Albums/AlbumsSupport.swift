import SwiftUI

// What the albums area's screens share: list rows, the website's wording for counts and date ranges, the album
// API's refusals in plain words, owner actions through the undo history, and the album cover.

/// One album in a list. The generated type is not Identifiable, and other areas list albums too, so the albums
/// area wraps it instead of adding a conformance to the API type.
struct AlbumsRow: Identifiable, Hashable {
    let album: API.AlbumOut
    var id: Int { album.id }
}

/// The albums area's words (the website's lib/albums.ts and the album pages).
enum AlbumsText {
    /// "November 2026" when the album spans one month, "Mar 2026 to Nov 2026" otherwise (albumDateRange).
    static func dateRange(oldest: String?, newest: String?) -> String {
        guard let a = ISODate.parse(oldest), let b = ISODate.parse(newest) else { return "" }
        let cal = Calendar.current
        if cal.component(.year, from: a) == cal.component(.year, from: b),
           cal.component(.month, from: a) == cal.component(.month, from: b) {
            return a.formatted(.dateTime.month(.wide).year())
        }
        return "\(a.formatted(.dateTime.month(.abbreviated).year())) to \(b.formatted(.dateTime.month(.abbreviated).year()))"
    }

    /// The line under an album's card: "12 items, Mar 2026 to Nov 2026".
    static func cardLine(_ album: API.AlbumOut) -> String {
        let range = dateRange(oldest: album.oldest, newest: album.newest)
        return Fmt.count(album.count, "item") + (range.isEmpty ? "" : ", \(range)")
    }

    /// The album page's facts: "12 items, Mar 2026 to Nov 2026, 1 private item in the vault".
    static func facts(_ album: API.AlbumOut) -> String {
        var parts = [Fmt.count(album.count, "item")]
        let range = dateRange(oldest: album.oldest, newest: album.newest)
        if !range.isEmpty { parts.append(range) }
        if album.vaultCount > 0 { parts.append(Fmt.count(album.vaultCount, "private item") + " in the vault") }
        return parts.joined(separator: ", ")
    }

    /// What VoiceOver reads for an album's card: the name first, then the card's line.
    static func cardLabel(_ album: API.AlbumOut) -> String {
        var text = "\(album.name), \(cardLine(album))"
        if album.vaultCount > 0 { text += ", \(Fmt.count(album.vaultCount, "private item")) in the vault" }
        return text
    }

    /// After adding to an album: "3 items added to Goa." (the website's bulk message).
    static func added(_ count: Int, to name: String) -> String {
        count > 0 ? "\(Fmt.count(count, "item")) added to \(name)." : "No items were added to \(name). They are in it already."
    }

    /// The album API's refusals in the website's words, else the app's usual explanation.
    static func error(_ error: Error, doing: String) -> String {
        if let api = error as? APIError, api.detail == "album_exists" {
            return "An album with this name exists already."
        }
        return ErrorText.describe(error, doing: doing)
    }
}

/// Owner actions of the albums area. Every write goes through the undo history, so a change the server recorded
/// offers Undo in its message; when the server changed nothing (adding items that are in the album already), the
/// message still shows, without Undo.
@MainActor
enum AlbumsActions {
    @discardableResult
    static func run<T>(_ label: String, _ body: () async throws -> T) async throws -> T {
        let before = UndoCenter.shared.entries.first?.id
        let value = try await UndoCenter.shared.perform(label, body)
        if UndoCenter.shared.entries.first?.id == before {
            Toaster.shared.ok(label)
        }
        return value
    }

    /// Like `run`, with the message made from the server's answer (the counts it returned). `label` names the
    /// action in the undo history.
    @discardableResult
    static func runCounting<T>(_ label: String, message: (T) -> String, _ body: () async throws -> T) async throws -> T {
        let before = UndoCenter.shared.entries.first?.id
        let value = try await UndoCenter.shared.perform(label, toast: false, body)
        let text = message(value)
        if let entry = UndoCenter.shared.entries.first, entry.id != before {
            let id = entry.id
            Toaster.shared.ok(text, action: .init(label: "Undo") { Task { await UndoCenter.shared.undo(id) } })
        } else {
            Toaster.shared.ok(text)
        }
        return value
    }
}

/// An album's cover, square: its picture, or the album symbol when it has none, with the vault count when it holds
/// private items (the website's AlbumCard).
struct AlbumsCover: View {
    let album: API.AlbumOut
    var cornerRadius: CGFloat = 16
    var maxPixel = 512
    var showsVaultCount = true

    var body: some View {
        Color(.secondarySystemBackground)
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                if let cover = album.coverUrl {
                    RemoteImage(ref: cover, maxPixel: maxPixel)
                } else {
                    Image(systemName: "photo.stack")
                        .font(.system(size: cornerRadius > 10 ? 34 : 18))
                        .foregroundStyle(.tertiary)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(alignment: .bottomTrailing) {
                if showsVaultCount && album.vaultCount > 0 {
                    Label("\(album.vaultCount)", systemImage: "lock.fill")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .foregroundStyle(.white)
                        .background(.black.opacity(0.6), in: Capsule())
                        .padding(6)
                }
            }
            .accessibilityHidden(true)
    }
}
