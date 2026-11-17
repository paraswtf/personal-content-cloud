import SwiftUI
import UIKit

// Share links, part of the albums area (docs/ARCHITECTURE.md section 26; the website's ShareModal, Settings >
// Sharing and lib/shares.ts). Other areas use SharingTarget with SharingCreateSheet(target:), and Settings opens
// SharingLinksScreen(); their names and parameters stay.

/// What a share link shows: one album, or items picked by the owner (`API.ShareCreate.kind`). Vault items are never
/// part of a link: the screens that offer Share leave it out in the vault.
enum SharingTarget: Hashable {
    case album(id: Int, title: String)
    case items(uids: [String])
}

/// The expiry choices of the share sheets (the website's EXPIRY_CHOICES).
enum SharingExpiry: Int, CaseIterable, Identifiable, Hashable {
    case never = 0
    case oneDay = 1
    case sevenDays = 7
    case thirtyDays = 30

    var id: Int { rawValue }

    /// The days to send, nil for never.
    var days: Int? { self == .never ? nil : rawValue }

    var label: String {
        switch self {
        case .never: return "Never"
        case .oneDay: return "After 1 day"
        case .sevenDays: return "After 7 days"
        case .thirtyDays: return "After 30 days"
        }
    }
}

/// A share link's state: live, past its expiry, or taken down.
enum SharingStatus: Equatable {
    case active, expired, revoked

    var label: String {
        switch self {
        case .active: return "Active"
        case .expired: return "Expired"
        case .revoked: return "Revoked"
        }
    }
}

/// One share link in a list (the generated type is not Identifiable).
struct SharingRow: Identifiable, Hashable {
    let share: API.ShareOut
    var id: Int { share.id }
}

/// The share links' words and addresses.
enum SharingText {
    /// "the album", "this item", "these 3 items".
    static func what(_ target: SharingTarget) -> String {
        switch target {
        case .album: return "the album"
        case .items(let uids): return uids.count == 1 ? "this item" : "these \(uids.count.formatted()) items"
        }
    }

    /// The link to send: the server's address and the share's path ("/s/<token>").
    static func url(_ share: API.ShareOut) -> URL? {
        APIConnection.current?.url(encodedPath: share.path)
    }

    static func status(_ share: API.ShareOut, now: Date = Date()) -> SharingStatus {
        if share.revokedAt != nil { return .revoked }
        if let ends = ISODate.parse(share.expiresAt), ends <= now { return .expired }
        return .active
    }

    /// "Never expires", "Expired", "Expires in 5 hours", "Expires in 7 days" (the website's expiryText).
    static func expiry(_ expiresAt: String?, now: Date = Date()) -> String {
        guard let expiresAt else { return "Never expires" }
        guard let ends = ISODate.parse(expiresAt) else { return "" }
        let seconds = ends.timeIntervalSince(now)
        if seconds <= 0 { return "Expired" }
        let hours = Int((seconds / 3600).rounded(.up))
        if hours < 48 { return "Expires in \(hours) \(hours == 1 ? "hour" : "hours")" }
        let days = Int((seconds / 86400).rounded(.up))
        return "Expires in \(days) days"
    }

    /// "Album: Goa (12 items)" or "3 items".
    static func describe(_ share: API.ShareOut) -> String {
        let count = Fmt.count(share.count, "item")
        switch share.kind {
        case .album:
            let name = share.album.map { ": \($0.name)" } ?? ""
            return "Album\(name) (\(count))"
        default:
            return count
        }
    }

    /// The line under a link in the list: what it shows, when it was made, its views and its expiry.
    static func details(_ share: API.ShareOut, now: Date = Date()) -> String {
        var text = "\(describe(share)). Created \(Fmt.date(share.createdAt)). \(Fmt.count(share.viewCount, "view"))"
        if let last = share.lastViewedAt { text += ", last \(Fmt.relative(last, now: now))" }
        text += ". "
        if status(share, now: now) == .revoked {
            text += "Taken down \(Fmt.date(share.revokedAt))."
        } else {
            text += expiry(share.expiresAt, now: now) + "."
        }
        if !share.allowDownload { text += " Downloads off." }
        return text
    }

    /// The share API's refusals in the website's words (shareErrorText), else `fallback` and the app's explanation.
    static func error(_ error: Error, fallback: String) -> String {
        if let api = error as? APIError {
            switch api.detail {
            case "vault_item": return "Private items cannot be shared. Take them out of the selection first."
            case "no_items": return "Nothing to share: the selection has no photos or videos."
            case "album_not_found": return "That album no longer exists."
            case "item_not_found": return "One of the items no longer exists."
            case "share_not_found": return "This link does not exist or was taken down."
            case "share_expired": return "This link has expired."
            default: break
            }
            if api.status == 0 { return "The server could not be reached." }
        }
        return "\(fallback) \(ErrorText.describe(error))"
    }

    /// Copies the link and says so.
    @MainActor
    static func copy(_ share: API.ShareOut) {
        guard let link = url(share) else {
            Toaster.shared.error("Could not copy. The server's address is not known.")
            return
        }
        UIPasteboard.general.string = link.absoluteString
        Toaster.shared.ok("Link copied.")
    }
}

/// A share link's state as a small coloured label.
struct SharingStatusBadge: View {
    let status: SharingStatus

    var body: some View {
        switch status {
        case .active: Chip(text: status.label, tint: .green)
        case .expired: Chip(text: status.label, tint: .orange)
        case .revoked: Chip(text: status.label, tint: .secondary)
        }
    }
}
