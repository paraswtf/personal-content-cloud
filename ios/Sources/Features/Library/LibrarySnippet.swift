import SwiftUI

/// A search snippet as plain text with its matched words marked (the website's components/library/Snippet.tsx). The
/// server escapes the text and wraps matched terms in <b></b> (search/service.py); <mark> and the raw \u{2} and
/// \u{3} markers are accepted too. The markers become marked runs, the basic entities are decoded, and nothing else
/// is read as markup: text read from pictures is untrusted.
enum LibrarySnippet {
    struct Part: Equatable {
        let text: String
        let hit: Bool
    }

    private static let openers = ["<mark>", "<b>", "\u{2}"]
    private static let markers = ["<mark>", "</mark>", "<b>", "</b>", "\u{2}", "\u{3}"]
    private static let entities: [(String, String)] = [
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&#x27;", "'"), ("&#39;", "'"),
    ]

    /// The snippet cut into runs, each matched or not, with the entities decoded.
    static func parts(_ text: String) -> [Part] {
        var out: [Part] = []
        var hit = false
        var rest = Substring(text)
        while !rest.isEmpty {
            var first: (range: Range<Substring.Index>, marker: String)?
            for marker in markers {
                if let r = rest.range(of: marker), first == nil || r.lowerBound < first!.range.lowerBound {
                    first = (r, marker)
                }
            }
            guard let found = first else {
                out.append(Part(text: decode(String(rest)), hit: hit))
                break
            }
            if found.range.lowerBound > rest.startIndex {
                out.append(Part(text: decode(String(rest[rest.startIndex..<found.range.lowerBound])), hit: hit))
            }
            hit = openers.contains(found.marker)
            rest = rest[found.range.upperBound...]
        }
        return out.filter { !$0.text.isEmpty }
    }

    /// The snippet as one piece of text: matched runs bold on the accent colour.
    static func attributed(_ text: String) -> AttributedString {
        var out = AttributedString()
        for part in parts(text) {
            var run = AttributedString(part.text)
            if part.hit {
                let bold: Font = .caption2.bold()
                let mark: Color = Color.accentColor.opacity(0.55)
                run.font = bold
                run.backgroundColor = mark
            }
            out += run
        }
        return out
    }

    /// The snippet without its markers, for VoiceOver.
    static func plain(_ text: String) -> String {
        parts(text).map(\.text).joined()
    }

    static func decode(_ text: String) -> String {
        guard text.contains("&") else { return text }
        var out = text
        for (entity, value) in entities where entity != "&amp;" {
            out = out.replacingOccurrences(of: entity, with: value)
        }
        // last, so "&amp;lt;" stays the text "&lt;"
        return out.replacingOccurrences(of: "&amp;", with: "&")
    }
}

/// What a library tile shows of an item beyond what any media item has: the flags, copies, document labels and,
/// for search hits, the snippet and the matched moment of a video.
protocol LibraryTileItem: MediaItem {
    var misclick: Bool { get }
    var forward: Bool { get }
    var dupCount: Int { get }
    var docType: String? { get }
    var pageCount: Int? { get }
    var filename: String { get }
    var category: API.Category { get }
    var snippetText: String? { get }
    var matchSeconds: Double? { get }
}

extension API.ItemCard: LibraryTileItem {
    var snippetText: String? { nil }
    var matchSeconds: Double? { nil }
}

extension API.SearchHit: LibraryTileItem {
    var snippetText: String? { snippet }
    var matchSeconds: Double? { kind == .video ? matchTimeS : nil }
}

extension LibraryTileItem {
    /// A document's tile shows its type or file extension (the website's isDocument).
    var isDocumentTile: Bool {
        mediaKind == .pdf || mediaKind == .document || (mediaKind == .image && docType != nil && category == .documents)
    }

    /// What VoiceOver reads for the tile, as the website's describe(): "Photo, Nov 10, 2026, favourite".
    var libraryLabel: String {
        var parts: [String] = []
        if docType != nil {
            parts.append(LibraryDocTypes.label(docType))
        } else if isLive && mediaKind == .image {
            parts.append("Live Photo")
        } else {
            parts.append(mediaKind.label)
        }
        if mediaKind == .video, let d = durationSeconds, d > 0 { parts.append(Fmt.duration(d)) }
        if let at = matchSeconds { parts.append("match at \(Fmt.duration(at))") }
        let date = Fmt.date(sortDateText)
        if !date.isEmpty { parts.append(date) }
        if isFavorite { parts.append("favourite") }
        if misclick { parts.append("possible mis-click") }
        if forward { parts.append("forward") }
        if dupCount > 0 { parts.append("\(dupCount) similar") }
        return parts.joined(separator: ", ")
    }
}
