import SwiftUI
import UIKit

/// The app's colour theme (the website's AppearanceTab and ThemeToggle, kept on this phone).
enum SettingsAppearanceChoice: String, CaseIterable, Identifiable {
    case system, light, dark

    /// Where the choice is kept (UserDefaults, on this phone only).
    static let storageKey = "pcc.appearance"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .system: return "System"
        case .light: return "Light"
        case .dark: return "Dark"
        }
    }

    var hint: String {
        switch self {
        case .system: return "Follows the device setting"
        case .light: return "Always light"
        case .dark: return "Always dark"
        }
    }

    var systemImage: String {
        switch self {
        case .system: return "iphone"
        case .light: return "sun.max"
        case .dark: return "moon"
        }
    }

    /// The windows' style: unspecified follows the phone.
    var style: UIUserInterfaceStyle {
        switch self {
        case .system: return .unspecified
        case .light: return .light
        case .dark: return .dark
        }
    }
}

/// Settings > Appearance: System, Light or Dark for this app.
struct SettingsAppearanceScreen: View {
    @AppStorage(SettingsAppearanceChoice.storageKey) private var choice: SettingsAppearanceChoice = .system

    var body: some View {
        List {
            Section {
                ForEach(SettingsAppearanceChoice.allCases) { option in
                    let selected = option == choice
                    Button {
                        choice = option
                    } label: {
                        HStack(spacing: 12) {
                            Image(systemName: option.systemImage)
                                .foregroundStyle(selected ? Color.accentColor : Color.secondary)
                                .frame(width: 28)
                                .accessibilityHidden(true)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(option.label)
                                    .foregroundStyle(.primary)
                                Text(option.hint)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if selected {
                                Image(systemName: "checkmark")
                                    .foregroundStyle(Color.accentColor)
                                    .accessibilityHidden(true)
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(selected ? .isSelected : [])
                    .accessibilityIdentifier("appearance.\(option.rawValue)")
                }
            } header: {
                Text("Theme")
            } footer: {
                Text("Saved on this phone. Other devices keep their own choice.")
            }
        }
        .navigationTitle("Appearance")
    }
}

extension View {
    /// Applies the theme chosen in Settings > Appearance to the app's windows, sheets included. The app's root
    /// (App.swift) calls it once.
    func settingsAppearance() -> some View {
        modifier(SettingsAppearanceModifier())
    }
}

/// Keeps the windows' style in step with the stored choice.
struct SettingsAppearanceModifier: ViewModifier {
    @AppStorage(SettingsAppearanceChoice.storageKey) private var choice: SettingsAppearanceChoice = .system

    func body(content: Content) -> some View {
        content
            .onAppear { Self.apply(choice) }
            .onChange(of: choice) { _, value in Self.apply(value) }
    }

    @MainActor
    static func apply(_ choice: SettingsAppearanceChoice) {
        for scene in UIApplication.shared.connectedScenes {
            guard let windowScene = scene as? UIWindowScene else { continue }
            for window in windowScene.windows {
                window.overrideUserInterfaceStyle = choice.style
            }
        }
    }
}
