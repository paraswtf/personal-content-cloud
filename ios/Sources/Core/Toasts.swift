import SwiftUI
import Observation
import UIKit

/// Short messages at the bottom of the screen: what was done (often with Undo) or what went wrong.
@MainActor
@Observable
final class Toaster {
    static let shared = Toaster()

    struct ToastAction {
        let label: String
        let run: () -> Void
    }

    struct Toast: Identifiable, Equatable {
        let id = UUID()
        let text: String
        let isError: Bool
        let action: ToastAction?

        static func == (a: Toast, b: Toast) -> Bool { a.id == b.id }
    }

    private(set) var current: Toast?
    @ObservationIgnored private var hideTask: Task<Void, Never>?

    func ok(_ text: String, action: ToastAction? = nil) {
        show(Toast(text: text, isError: false, action: action))
    }

    func error(_ text: String, action: ToastAction? = nil) {
        show(Toast(text: text, isError: true, action: action))
    }

    /// Shows what went wrong in plain words; a cancelled request shows nothing.
    func error(_ error: Error, doing: String? = nil) {
        if error is CancellationError { return }
        if let u = error as? URLError, u.code == .cancelled { return }
        self.error(ErrorText.describe(error, doing: doing))
    }

    func dismiss(_ id: UUID? = nil) {
        if id == nil || current?.id == id { current = nil }
    }

    private func show(_ toast: Toast) {
        current = toast
        UIAccessibility.post(notification: .announcement, argument: toast.text)
        hideTask?.cancel()
        let seconds: UInt64 = (toast.action != nil || toast.isError) ? 6 : 3
        hideTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: seconds * 1_000_000_000)
            guard !Task.isCancelled else { return }
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

/// The toast at the bottom of the screen, above everything else.
struct ToastHost: View {
    var body: some View {
        let toaster = Toaster.shared
        VStack {
            Spacer()
            if let toast = toaster.current {
                HStack(spacing: 12) {
                    Image(systemName: toast.isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                        .foregroundStyle(toast.isError ? Color.orange : Color.green)
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
                .padding(.bottom, 64)
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .onTapGesture { toaster.dismiss(toast.id) }
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("toast")
            }
        }
        .animation(.easeOut(duration: 0.2), value: toaster.current)
        .allowsHitTesting(toaster.current != nil)
    }
}

extension View {
    /// Shows the app's toasts over this view (once, at the root).
    func toastHost() -> some View {
        overlay { ToastHost() }
    }
}
