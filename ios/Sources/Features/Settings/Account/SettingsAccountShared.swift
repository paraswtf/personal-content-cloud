import SwiftUI
import UIKit

/// Ends the typing: the keyboard goes away so what the form says next is in view.
@MainActor
func settingsAccountEndEditing() {
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
}

/// The words and small rules of the account side of Settings, as the website has them
/// (components/shell/settings/common.tsx, AuthCard.tsx, lib/auth.ts).
enum SettingsAccountText {
    static let passwordMin = 10
    static let passwordMax = 200
    static let deviceNameMax = 80
    static let passkeyNameMax = 60

    /// What the website says when the vault is still locked after it was opened once.
    static let vaultStillLocked = "Unlock the vault first: enter your password or use a passkey."

    /// "Chrome on Android", "Safari on iPhone", from a user agent string (best effort, the website's describeAgent).
    static func describeAgent(_ ua: String?) -> String {
        guard let ua, !ua.isEmpty else { return "Unknown device" }
        func has(_ pattern: String, ignoringCase: Bool = false) -> Bool {
            let options: String.CompareOptions = ignoringCase ? [.regularExpression, .caseInsensitive] : [.regularExpression]
            return ua.range(of: pattern, options: options) != nil
        }
        let browser: String
        if has("Edg/") {
            browser = "Edge"
        } else if has("OPR/|Opera") {
            browser = "Opera"
        } else if has("SamsungBrowser") {
            browser = "Samsung Internet"
        } else if has("Firefox/|FxiOS") {
            browser = "Firefox"
        } else if has("CriOS|Chrome/") {
            browser = "Chrome"
        } else if has("Safari/") {
            browser = "Safari"
        } else if has("curl|python|httpx|okhttp", ignoringCase: true) {
            browser = "Script"
        } else {
            browser = "Browser"
        }
        let os: String
        if has("iPhone") {
            os = "iPhone"
        } else if has("iPad") {
            os = "iPad"
        } else if has("Android") {
            os = "Android"
        } else if has("CrOS") {
            os = "ChromeOS"
        } else if has("Mac OS X|Macintosh") {
            os = "Mac"
        } else if has("Windows") {
            os = "Windows"
        } else if has("Linux") {
            os = "Linux"
        } else {
            os = ""
        }
        return os.isEmpty ? browser : "\(browser) on \(os)"
    }

    /// A phone's browser (the website draws a phone for it, a laptop for the rest).
    static func isPhone(_ ua: String?) -> Bool {
        guard let ua else { return false }
        return ua.range(of: "iPhone|Android|Mobile", options: .regularExpression) != nil
    }

    /// A device's name, or what its user agent says when it has none.
    static func deviceLabel(_ device: API.SessionOut) -> String {
        if let name = device.deviceName, !name.isEmpty { return name }
        return describeAgent(device.userAgent)
    }

    /// Runs of spaces made one and the ends trimmed, as the website cleans names, then cut to `max` characters.
    static func cleanName(_ text: String, max: Int) -> String {
        let collapsed = text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        return String(collapsed.prefix(max))
    }

    /// How strong a new password looks (the website's passwordStrength): a score of 0 to 3 and its words.
    struct Strength: Equatable {
        let score: Int
        let label: String
    }

    static func passwordStrength(_ password: String) -> Strength {
        if password.isEmpty { return Strength(score: 0, label: "") }
        var classes = 0
        for pattern in ["[a-z]", "[A-Z]", "[0-9]", "[^A-Za-z0-9]"]
        where password.range(of: pattern, options: .regularExpression) != nil {
            classes += 1
        }
        let unique = Set(password).count
        let length = password.count
        if length < 10 { return Strength(score: 0, label: "Too short: use at least 10 characters.") }
        if unique < 5 || (length < 12 && classes < 2) {
            return Strength(score: 1, label: "Weak: add length or mix in other kinds of characters.")
        }
        if length >= 16 || (length >= 12 && classes >= 3) { return Strength(score: 3, label: "Strong.") }
        return Strength(score: 2, label: "Fair: a few more characters make it stronger.")
    }

    /// The error under "New password" once the form was sent (the website's checks, made before anything is sent).
    static func newPasswordError(current: String, new: String) -> String? {
        if new.count < passwordMin { return "Use at least \(passwordMin) characters." }
        if new.count > passwordMax { return "Keep it under \(passwordMax) characters." }
        if !new.isEmpty && new == current { return "The new password is the same as the current one." }
        return nil
    }

    /// The error under "New password again".
    static func matchError(new: String, again: String) -> String? {
        again == new ? nil : "The two passwords do not match."
    }

    /// The server refused because the vault is locked (403 `vault_locked`): adding a device or a passkey, or
    /// removing a passkey, needs it open.
    static func isVaultLocked(_ error: Error) -> Bool {
        guard let api = error as? APIError else { return false }
        return api.status == 403 && api.detail == "vault_locked"
    }

    /// "203.0.113.4, signed in 3 Nov 2026, last active 5 Nov 2026 at 10:30".
    static func sessionLine(_ session: API.SessionOut) -> String {
        var ip = ""
        if let address = session.ip, !address.isEmpty { ip = "\(address), " }
        return "\(ip)signed in \(Fmt.date(session.createdAt)), last active \(Fmt.date(session.lastSeenAt, withTime: true))"
    }

    /// "Added 3 Nov 2026, last used 5 min ago".
    static func deviceLine(_ device: API.SessionOut) -> String {
        "Added \(Fmt.date(device.createdAt)), last used \(Fmt.relative(device.lastSeenAt))"
    }

    /// "Added 3 Nov 2026, last used 5 Nov 2026 at 10:30", or ", not used yet".
    static func passkeyLine(_ key: API.PasskeyOut) -> String {
        let used = key.lastUsedAt.map { ", last used \(Fmt.date($0, withTime: true))" } ?? ", not used yet"
        return "Added \(Fmt.date(key.createdAt))\(used)"
    }
}

/// An action the server refused because the vault is locked, run again once after the vault was opened.
enum SettingsAccountPending: Identifiable, Equatable {
    case addDevice(name: String)
    case removePasskey(API.PasskeyOut)

    var id: String {
        switch self {
        case .addDevice(let name): return "device:\(name)"
        case .removePasskey(let key): return "passkey:\(key.id)"
        }
    }
}

/// A password field with a button that shows or hides what was typed (the website's PasswordInput).
struct SettingsAccountPasswordField: View {
    let title: String
    @Binding var text: String
    var isNew = false
    let identifier: String

    @State private var shown = false

    var body: some View {
        HStack(spacing: 8) {
            Group {
                if shown {
                    TextField(title, text: $text)
                } else {
                    SecureField(title, text: $text)
                }
            }
            .textContentType(isNew ? .newPassword : .password)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .accessibilityIdentifier(identifier)
            Button {
                shown.toggle()
            } label: {
                Image(systemName: shown ? "eye.slash" : "eye")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(shown ? "Hide the password" : "Show the password")
        }
    }
}

/// The three bars and the words under a new password (the website's StrengthMeter).
struct SettingsAccountStrengthMeter: View {
    let password: String

    var body: some View {
        let strength = SettingsAccountText.passwordStrength(password)
        if !password.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 4) {
                    ForEach(1...3, id: \.self) { n in
                        Capsule()
                            .fill(strength.score >= n ? tone(strength.score) : Color.secondary.opacity(0.2))
                            .frame(height: 4)
                    }
                }
                .accessibilityHidden(true)
                Text(strength.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 2)
        }
    }

    private func tone(_ score: Int) -> Color {
        score >= 3 ? .green : (score == 2 ? .orange : .red)
    }
}

/// A line of error text under a field.
struct SettingsAccountFieldError: View {
    let text: String?

    var body: some View {
        if let text {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: "exclamationmark.circle")
                    .accessibilityHidden(true)
                Text(text)
            }
            .font(.footnote)
            .foregroundStyle(.red)
        }
    }
}
