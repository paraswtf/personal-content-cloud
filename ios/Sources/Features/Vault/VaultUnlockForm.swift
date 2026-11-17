import SwiftUI

/// The vault's unlock form (the website's UnlockForm): the account's password. Full size on the vault screen while
/// it is locked; compact in VaultUnlockSheet. A passkey cannot be used from the app (a passkey belongs to the
/// server's own web address, which an app can only use with an associated domain made for that address), so with
/// passkeys on the account the form points to the website for them, as the website does on another address.
struct VaultUnlockForm: View {
    var compact = false
    var description: String?
    var onUnlocked: () -> Void = {}

    @State private var password = ""
    @State private var error: String?
    @State private var busy = false
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 16) {
            if !compact {
                Image(systemName: "lock.shield")
                    .font(.system(size: 30, weight: .semibold))
                    .frame(width: 64, height: 64)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
                    .accessibilityHidden(true)
                Text(VaultText.lockedTitle)
                    .font(.title2.bold())
                    .multilineTextAlignment(.center)
                    .accessibilityAddTraits(.isHeader)
            }
            if let text = description ?? (compact ? nil : VaultText.lockedText) {
                Text(text)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(compact ? .leading : .center)
                    .frame(maxWidth: .infinity, alignment: compact ? .leading : .center)
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("Password")
                    .font(.subheadline.weight(.medium))
                    .accessibilityHidden(true)
                SecureField("Password", text: $password)
                    .textContentType(.password)
                    .submitLabel(.go)
                    .focused($focused)
                    .onSubmit(submit)
                    .padding(.horizontal, 16)
                    .frame(height: 44)
                    .background(Color(.secondarySystemBackground), in: Capsule())
                    .accessibilityLabel("Password")
                    .accessibilityIdentifier("vault.password")
                if let error {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .accessibilityIdentifier("vault.error")
                }
            }
            Button(action: submit) {
                HStack(spacing: 8) {
                    if busy {
                        ProgressView()
                    } else {
                        Image(systemName: "lock.open")
                    }
                    Text("Unlock")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(password.isEmpty || busy)
            .accessibilityIdentifier("vault.unlock")
            if VaultSession.shared.passkeys > 0 {
                passkeyNote
            }
        }
        .frame(maxWidth: 420)
        .onAppear { if compact { focused = true } }
    }

    /// The website's note for a place where its passkeys cannot be used: the password here, or a passkey there.
    @ViewBuilder private var passkeyNote: some View {
        if let base = APIConnection.current?.base {
            VStack(spacing: 4) {
                Text("Use your password, or a passkey on the website at")
                    .accessibilityIdentifier("vault.passkeyNote")
                Link(base.host ?? base.absoluteString, destination: base.appendingPathComponent("vault"))
            }
            .font(.footnote)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
        }
    }

    private func submit() {
        guard !password.isEmpty, !busy else { return }
        error = nil
        busy = true
        let secret = password
        Task {
            defer { busy = false }
            do {
                try await VaultSession.shared.unlock(password: secret)
                password = ""
                onUnlocked()
            } catch is CancellationError {
            } catch {
                self.error = VaultText.unlockError(error)
                // a wrong password is typed again, not edited
                if let api = error as? APIError, api.status == 401 { password = "" }
                focused = true
            }
        }
    }
}
