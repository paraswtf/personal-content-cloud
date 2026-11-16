import SwiftUI

// The account side of Settings (components/shell/settings: AccountTab, PasskeysTab, SessionsTab, WalletTab,
// AppearanceTab). Settings opens these five screens by these names. Placeholders until the area builds them.

struct SettingsAccountScreen: View {
    var body: some View { PlaceholderScreen(title: "Account", systemImage: "person.crop.circle") }
}

struct SettingsPasskeysScreen: View {
    var body: some View { PlaceholderScreen(title: "Passkeys", systemImage: "person.badge.key") }
}

struct SettingsSessionsScreen: View {
    var body: some View { PlaceholderScreen(title: "Sessions and devices", systemImage: "laptopcomputer.and.iphone") }
}

struct SettingsWalletScreen: View {
    var body: some View { PlaceholderScreen(title: "Wallet", systemImage: "wallet.pass") }
}

struct SettingsAppearanceScreen: View {
    var body: some View { PlaceholderScreen(title: "Appearance", systemImage: "circle.lefthalf.filled") }
}
