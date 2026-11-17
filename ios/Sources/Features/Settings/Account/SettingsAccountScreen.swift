import SwiftUI

/// Settings > Account (the website's AccountTab): the username, change password and sign out everywhere.
struct SettingsAccountScreen: View {
    @EnvironmentObject private var session: AppSession

    @State private var username: String?
    @State private var loadError: Error?
    @State private var current = ""
    @State private var newPassword = ""
    @State private var again = ""
    @State private var touched = false
    @State private var busy = false
    @State private var formError: String?
    @State private var askAll = false
    @State private var allBusy = false

    private var newError: String? {
        touched ? SettingsAccountText.newPasswordError(current: current, new: newPassword) : nil
    }

    private var matchError: String? {
        touched ? SettingsAccountText.matchError(new: newPassword, again: again) : nil
    }

    var body: some View {
        Form {
            Section {
                HStack {
                    Text("Username")
                    Spacer()
                    Text(username ?? (loadError == nil ? "" : "-"))
                        .fontWeight(.semibold)
                        .accessibilityIdentifier("account.username")
                }
                if let loadError {
                    Text(ErrorText.describe(loadError, doing: "load the account"))
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            } header: {
                Text("Account")
            } footer: {
                Text("This cloud has one account. It owns everything stored here.")
            }

            Section {
                SettingsAccountPasswordField(title: "Current password", text: $current, identifier: "account.current")
                if touched && current.isEmpty {
                    SettingsAccountFieldError(text: "Enter the current password.")
                }
                SettingsAccountPasswordField(title: "New password", text: $newPassword, isNew: true,
                                             identifier: "account.new")
                SettingsAccountStrengthMeter(password: newPassword)
                SettingsAccountFieldError(text: newError)
                SettingsAccountPasswordField(title: "New password again", text: $again, isNew: true,
                                             identifier: "account.again")
                    .submitLabel(.go)
                    .onSubmit { Task { await submit() } }
                SettingsAccountFieldError(text: matchError)
                SettingsAccountFieldError(text: formError)
                Button {
                    Task { await submit() }
                } label: {
                    HStack {
                        Text("Change password")
                        if busy {
                            Spacer()
                            ProgressView()
                        }
                    }
                }
                .disabled(busy)
                .accessibilityIdentifier("account.change")
            } header: {
                Text("Change password")
            } footer: {
                Text("Every other signed-in browser and device is signed out and signs in again with the new password. This phone stays signed in.")
            }

            Section {
                Button(role: .destructive) {
                    askAll = true
                } label: {
                    HStack {
                        Label("Sign out everywhere", systemImage: "rectangle.portrait.and.arrow.right")
                        if allBusy {
                            Spacer()
                            ProgressView()
                        }
                    }
                }
                .disabled(allBusy)
                .accessibilityIdentifier("account.signOutAll")
            } header: {
                Text("Sign out everywhere")
            } footer: {
                Text("Ends every session, including this one. Use it if a device was lost or someone else may have signed in.")
            }
        }
        .navigationTitle("Account")
        .task { await load() }
        .refreshable { await load() }
        .confirmAction("Sign out everywhere", isPresented: $askAll, actionTitle: "Sign out everywhere",
                       message: "Every device, this phone included, will need the password or a new device token to sign in again.") {
            Task { await signOutEverywhere() }
        }
    }

    private func load() async {
        do {
            let state = try await Cloud.api().getState()
            username = state.username
            loadError = nil
        } catch {
            if error is CancellationError { return }
            loadError = error
        }
    }

    private func submit() async {
        settingsAccountEndEditing()
        touched = true
        guard !busy, !current.isEmpty,
              SettingsAccountText.newPasswordError(current: current, new: newPassword) == nil,
              SettingsAccountText.matchError(new: newPassword, again: again) == nil else { return }
        busy = true
        formError = nil
        defer { busy = false }
        let body = API.PasswordChangeIn(currentPassword: current, newPassword: newPassword)
        do {
            try await UndoCenter.shared.perform("Password changed", toast: false) {
                _ = try await Cloud.api().changePassword(body: body)
            }
            current = ""
            newPassword = ""
            again = ""
            touched = false
            Toaster.shared.ok("Password changed.")
        } catch let error as APIError where error.detail == "login_required" {
            session.signOut()
        } catch let error as APIError where [400, 401, 403].contains(error.status) && error.detail != "csrf" {
            formError = "The current password is not right."
        } catch let error as APIError where error.status == 429 {
            formError = "Too many attempts. Wait a few minutes, then try again."
        } catch {
            if error is CancellationError { return }
            formError = ErrorText.describe(error, doing: "change the password")
        }
    }

    /// Ends every session on the server, this phone's device token too, then signs this phone out.
    private func signOutEverywhere() async {
        allBusy = true
        defer { allBusy = false }
        do {
            let state = try await UndoCenter.shared.perform("Signed out everywhere", toast: false) {
                try await Cloud.api().logoutAll()
            }
            if state.authenticated {
                Toaster.shared.ok("Every other session was signed out.")
            } else {
                session.signOut()
            }
        } catch let error as APIError where error.status == 401 {
            session.signOut()
        } catch {
            Toaster.shared.error(error, doing: "sign out everywhere")
        }
    }
}
