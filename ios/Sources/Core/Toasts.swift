import SwiftUI
import Observation
import UIKit

/// Short messages at the bottom of the screen: what was done (often with Undo), what went wrong, or a plain note.
/// Up to four show at once, the newest at the bottom (the website's Toast.tsx).
@MainActor
@Observable
final class Toaster {
    static let shared = Toaster()
    /// The most messages on the screen at once; a fifth pushes the oldest out.
    static let maxShown = 4

    struct ToastAction {
        let label: String
        let run: () -> Void
    }

    /// Success, error or a plain note (the website's ok, error and info tones).
    enum Tone: Equatable {
        case ok, error, info
    }

    struct Toast: Identifiable, Equatable {
        let id = UUID()
        let text: String
        let tone: Tone
        let action: ToastAction?

        var isError: Bool { tone == .error }

        static func == (a: Toast, b: Toast) -> Bool { a.id == b.id }
    }

    /// The messages on the screen, oldest first.
    private(set) var toasts: [Toast] = []

    /// The newest message.
    var current: Toast? { toasts.last }

    func ok(_ text: String, action: ToastAction? = nil) {
        show(Toast(text: text, tone: .ok, action: action))
    }

    func error(_ text: String, action: ToastAction? = nil) {
        show(Toast(text: text, tone: .error, action: action))
    }

    /// A plain note that is neither a success nor a failure ("The vault is locked.").
    func info(_ text: String, action: ToastAction? = nil) {
        show(Toast(text: text, tone: .info, action: action))
    }

    /// Shows what went wrong in plain words; a cancelled request shows nothing.
    func error(_ error: Error, doing: String? = nil) {
        if error is CancellationError { return }
        if let u = error as? URLError, u.code == .cancelled { return }
        self.error(ErrorText.describe(error, doing: doing))
    }

    /// Takes one message away, or all of them when `id` is nil.
    func dismiss(_ id: UUID? = nil) {
        if let id {
            toasts.removeAll { $0.id == id }
        } else {
            toasts.removeAll()
        }
    }

    /// How long a message stays, as the website has it: 5 s, 8 s with a button, 9 s for an error and 12 s for an
    /// error with a button.
    nonisolated static func seconds(tone: Tone, hasAction: Bool) -> Double {
        switch tone {
        case .error: return hasAction ? 12 : 9
        case .ok, .info: return hasAction ? 8 : 5
        }
    }

    private func show(_ toast: Toast) {
        toasts = Array(toasts.suffix(Self.maxShown - 1)) + [toast]
        UIAccessibility.post(notification: .announcement, argument: toast.text)
        let seconds = Self.seconds(tone: toast.tone, hasAction: toast.action != nil)
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            self?.dismiss(toast.id)
        }
    }
}

/// Error messages in the app's words (the same codes the website explains, docs/ARCHITECTURE.md section 13).
enum ErrorText {
    static func describe(_ error: Error, doing: String? = nil) -> String {
        let prefix = doing.map { "Could not \($0). " } ?? ""
        if let api = error as? APIError {
            return prefix + describe(api)
        }
        if let u = error as? URLError {
            switch u.code {
            case .notConnectedToInternet, .networkConnectionLost, .dataNotAllowed:
                return prefix + "There is no connection. Try again when the phone is online."
            case .timedOut:
                return prefix + "The server did not answer in time."
            case .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed:
                return prefix + "The server could not be reached. Check its address."
            case .secureConnectionFailed, .serverCertificateUntrusted, .serverCertificateHasBadDate,
                 .serverCertificateNotYetValid, .serverCertificateHasUnknownRoot:
                return prefix + "The secure connection to the server failed."
            case .badURL, .unsupportedURL:
                return prefix + "The address is not valid."
            default:
                return prefix + sentence(u.localizedDescription)
            }
        }
        if error is DecodingError {
            return prefix + "The server's answer could not be read. The app may need an update."
        }
        return prefix + sentence(error.localizedDescription)
    }

    static func describe(_ api: APIError) -> String {
        switch (api.status, api.detail) {
        case (0, _): return sentence(api.detail)
        case (401, "invalid_credentials"): return "The username or password is wrong."
        case (401, _): return "The server no longer accepts this device. Sign in again."
        case (403, "vault_locked"): return "Unlock the vault to see this."
        case (403, _): return "The server refused this."
        case (404, _): return "It is no longer there."
        case (409, "setup_required"): return "This server has no account yet. Set it up on the website first."
        case (413, _): return "The file is larger than the server accepts."
        case (429, _): return "Too many tries. Wait a few minutes and try again."
        case (500...599, _): return "The server had a problem (\(api.status)). Try again in a moment."
        default:
            let text = sentence(api.detail.replacingOccurrences(of: "_", with: " "))
            return text.isEmpty ? "The server answered \(api.status)." : text
        }
    }

    /// "it worked" becomes "It worked." (one trailing full stop, a capital first letter).
    static func sentence(_ text: String) -> String {
        var t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        while let last = t.last, last == "." || last == " " { t.removeLast() }
        guard let first = t.first else { return "" }
        return first.uppercased() + String(t.dropFirst()) + "."
    }
}

/// The messages at the bottom of the screen, above everything else, the newest lowest. Only the newest carries the
/// identifier "toast" (the screen tests read it); the older ones are "toast.older".
struct ToastHost: View {
    var body: some View {
        let toaster = Toaster.shared
        let toasts = toaster.toasts
        VStack(spacing: 8) {
            Spacer()
            ForEach(toasts) { toast in
                ToastRow(toast: toast, newest: toast.id == toasts.last?.id)
            }
        }
        .padding(.bottom, 64)
        .animation(.easeOut(duration: 0.2), value: toasts.map(\.id))
        .allowsHitTesting(!toasts.isEmpty)
    }
}

/// One message: its tone's icon, the text and its button.
struct ToastRow: View {
    let toast: Toaster.Toast
    let newest: Bool

    var body: some View {
        let toaster = Toaster.shared
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundStyle(color)
                .accessibilityHidden(true)
            Text(toast.text)
                .font(.subheadline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
            if let action = toast.action {
                Button(action.label) {
                    toaster.dismiss(toast.id)
                    action.run()
                }
                .font(.subheadline.weight(.semibold))
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .shadow(color: .black.opacity(0.15), radius: 8, y: 2)
        .padding(.horizontal, 16)
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .onTapGesture { toaster.dismiss(toast.id) }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier(newest ? "toast" : "toast.older")
    }

    private var icon: String {
        switch toast.tone {
        case .ok: return "checkmark.circle.fill"
        case .error: return "exclamationmark.triangle.fill"
        case .info: return "info.circle.fill"
        }
    }

    private var color: Color {
        switch toast.tone {
        case .ok: return .green
        case .error: return .orange
        case .info: return .secondary
        }
    }
}

extension View {
    /// Shows the app's toasts over this view (once, at the root).
    func toastHost() -> some View {
        overlay { ToastHost() }
    }
}
