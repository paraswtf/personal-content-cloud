import Foundation

/// The server's timestamps: "2026-11-16T19:18:17.000Z" (UTC with milliseconds), sometimes with microseconds or an
/// offset, sometimes without a zone (read as local time, as the website's Date does), or a plain date.
enum ISODate {
    private static let withFraction: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private static let plain: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    private static let local: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = .current
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return f
    }()

    private static let dateOnly: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = .current
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()

    static func parse(_ text: String?) -> Date? {
        guard let raw = text?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return nil }
        // Fractions longer than milliseconds (Python's microseconds) are cut to three digits.
        let t = raw.replacingOccurrences(of: #"\.(\d{3})\d+"#, with: ".$1", options: .regularExpression)
        if let d = withFraction.date(from: t) ?? plain.date(from: t) { return d }
        let noFraction = t.replacingOccurrences(of: #"\.\d+$"#, with: "", options: .regularExpression)
        if let d = local.date(from: noFraction) { return d }
        return dateOnly.date(from: String(t.prefix(10)))
    }
}

/// Display formatting, matching the website's lib/format.ts. Dates show in the phone's time zone and locale.
enum Fmt {
    static func bytes(_ n: Int64?) -> String {
        guard let n else { return "-" }
        let units = ["B", "KB", "MB", "GB", "TB"]
        var v = Double(n)
        var i = 0
        while v >= 1024 && i < units.count - 1 {
            v /= 1024
            i += 1
        }
        let number = (v >= 100 || i == 0) ? String(Int(v.rounded())) : String(format: "%.1f", v)
        return "\(number) \(units[i])"
    }

    static func bytes(_ n: Int?) -> String { bytes(n.map { Int64($0) }) }

    /// 75 -> "1:15", 3725 -> "1:02:05".
    static func duration(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite else { return "" }
        let s = max(0, Int(seconds.rounded()))
        let h = s / 3600, m = (s % 3600) / 60, sec = s % 60
        return h > 0 ? String(format: "%d:%02d:%02d", h, m, sec) : String(format: "%d:%02d", m, sec)
    }

    /// "7 Nov 2026", with the time when asked.
    static func date(_ iso: String?, withTime: Bool = false) -> String {
        guard let d = ISODate.parse(iso) else { return "" }
        return date(d, withTime: withTime)
    }

    static func date(_ d: Date, withTime: Bool = false) -> String {
        withTime
            ? d.formatted(.dateTime.year().month(.abbreviated).day().hour().minute())
            : d.formatted(.dateTime.year().month(.abbreviated).day())
    }

    /// "1 item", "1,204 items".
    static func count(_ n: Int, _ singular: String, _ plural: String? = nil) -> String {
        "\(n.formatted()) \(n == 1 ? singular : (plural ?? singular + "s"))"
    }

    /// The tail of a naming message when naming a face made guesses of the other faces the app had grouped with it
    /// in the background (docs/ARCHITECTURE.md section 10): " 2 faces that look alike wait in Review." (or "").
    static func guessedTail(_ guessed: Int?) -> String {
        guard let guessed, guessed > 0 else { return "" }
        let verb = guessed == 1 ? "looks alike waits" : "look alike wait"
        return " \(count(guessed, "face")) that \(verb) in Review."
    }

    /// The tail of a merge message: " 3 look-alikes came along." (or "").
    static func followedTail(_ followed: Int?) -> String {
        guard let followed, followed > 0 else { return "" }
        return " \(count(followed, "look-alike")) came along."
    }

    /// "just now", "5 min ago", "3 h ago", "2 days ago", then a date.
    static func relative(_ iso: String?, now: Date = Date()) -> String {
        guard let t = ISODate.parse(iso) else { return "" }
        let s = max(0, Int(now.timeIntervalSince(t).rounded()))
        if s < 45 { return "just now" }
        let m = Int((Double(s) / 60).rounded())
        if m < 60 { return "\(m) min ago" }
        let h = Int((Double(m) / 60).rounded())
        if h < 24 { return "\(h) h ago" }
        let d = Int((Double(h) / 24).rounded())
        if d < 7 { return "\(d) \(d == 1 ? "day" : "days") ago" }
        return date(t)
    }

    /// "in 8 h", "in 25 min", "in 2 days"; "now" once it has come.
    static func until(_ iso: String?, now: Date = Date()) -> String {
        guard let t = ISODate.parse(iso) else { return "" }
        let s = Int(t.timeIntervalSince(now).rounded())
        if s < 45 { return "now" }
        let m = Int((Double(s) / 60).rounded())
        if m < 60 { return "in \(m) min" }
        let h = Int((Double(m) / 60).rounded())
        if h < 24 { return "in \(h) h" }
        let d = Int((Double(h) / 24).rounded())
        return "in \(d) \(d == 1 ? "day" : "days")"
    }

    /// "Today", "Yesterday", or "Mon, 3 Nov 2026".
    static func dayHeading(_ iso: String?, now: Date = Date()) -> String {
        guard let d = ISODate.parse(iso) else { return "" }
        let cal = Calendar.current
        if cal.isDate(d, inSameDayAs: now) { return "Today" }
        if let y = cal.date(byAdding: .day, value: -1, to: now), cal.isDate(d, inSameDayAs: y) { return "Yesterday" }
        return d.formatted(.dateTime.weekday(.abbreviated).day().month(.abbreviated).year())
    }

    /// "2026-11" -> "November 2026".
    static func monthHeading(_ key: String) -> String {
        let parts = key.split(separator: "-")
        guard parts.count >= 2, let y = Int(parts[0]), let m = Int(parts[1]),
              let d = Calendar.current.date(from: DateComponents(year: y, month: m, day: 15)) else { return key }
        return d.formatted(.dateTime.month(.wide).year())
    }

    /// "18.5204° N, 73.8567° E".
    static func coords(lat: Double, lon: Double) -> String {
        func part(_ v: Double, _ pos: String, _ neg: String) -> String {
            String(format: "%.4f° %@", abs(v), v >= 0 ? pos : neg)
        }
        return "\(part(lat, "N", "S")), \(part(lon, "E", "W"))"
    }

    /// 0.861 -> "86%".
    static func percent(_ v: Double?, digits: Int = 0) -> String {
        guard let v, v.isFinite else { return "" }
        return String(format: "%.\(digits)f%%", v * 100)
    }

    /// 0.861 -> "0.86".
    static func score(_ v: Double?) -> String {
        guard let v, v.isFinite else { return "" }
        return String(format: "%.2f", v)
    }

    /// "2026-11-07" (a plain date, no time zone shift) -> "7 Nov 2026"; other text is returned as it is.
    static func plainDate(_ value: String?) -> String {
        guard let value, !value.isEmpty else { return "" }
        let parts = value.split(separator: "-")
        if value.count == 10, parts.count == 3, let y = Int(parts[0]), let m = Int(parts[1]), let d = Int(parts[2]),
           let date = Calendar.current.date(from: DateComponents(year: y, month: m, day: d)) {
            return date.formatted(.dateTime.day().month(.abbreviated).year())
        }
        let formatted = date(value)
        return formatted.isEmpty ? value : formatted
    }

    /// "3 min", "1 h 5 min".
    static func minutes(_ min: Double) -> String {
        let m = max(0, Int(min.rounded(.up)))
        if m < 60 { return "\(m) min" }
        let h = m / 60
        return m % 60 == 0 ? "\(h) h" : "\(h) h \(m % 60) min"
    }

    /// "14:05" for seconds left (the website's vault badge, mmss): whole minutes, then two-digit seconds.
    static func clock(_ seconds: Int) -> String {
        let s = max(0, seconds)
        return "\(s / 60):" + String(format: "%02d", s % 60)
    }

    /// "4032 x 3024 (12.2 MP)".
    static func resolution(_ w: Int?, _ h: Int?) -> String {
        guard let w, let h, w > 0, h > 0 else { return "" }
        let mp = Double(w * h) / 1e6
        return mp >= 1 ? String(format: "%d x %d (%.1f MP)", w, h, mp) : "\(w) x \(h)"
    }

    /// Two letters for an avatar.
    static func initials(_ name: String?) -> String {
        let parts = (name ?? "").split(whereSeparator: { $0.isWhitespace })
        guard let first = parts.first?.first else { return "?" }
        let last: String = parts.count > 1 ? String(parts[parts.count - 1].prefix(1)) : ""
        return (String(first) + last).uppercased()
    }

    /// "dog" -> "Dog", "guinea_pig" -> "Guinea pig".
    static func speciesLabel(_ species: String?) -> String {
        guard let species, !species.isEmpty else { return "Pet" }
        let words = species.replacingOccurrences(of: "_", with: " ")
        return String(words.prefix(1)).uppercased() + String(words.dropFirst())
    }

    /// A person's or pet's name for display. Unnamed people are not shown in the app, so this is only a
    /// fallback for a row that slipped through.
    static func personName(_ name: String?, kind: String? = nil, species: String? = nil) -> String {
        if let n = name?.trimmingCharacters(in: .whitespacesAndNewlines), !n.isEmpty { return n }
        if kind == "pet" { return species.map { "Unnamed \(speciesLabel($0).lowercased())" } ?? "Unnamed pet" }
        return "Not named"
    }

    /// True when a name is a real one (not empty): the app shows only named people.
    static func isNamed(_ name: String?) -> Bool {
        !(name?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
    }
}
