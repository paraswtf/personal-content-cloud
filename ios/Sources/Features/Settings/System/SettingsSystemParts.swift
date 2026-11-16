import SwiftUI

// Small pieces the system side of Settings shares (the website's Badge, Dot, Stat and Section header).

extension SettingsTone {
    var color: Color {
        switch self {
        case .ok: return .green
        case .warn: return .orange
        case .danger: return .red
        case .idle: return .gray
        }
    }
}

/// A short state word in a coloured capsule ("GPU in use", "Idle").
struct SettingsBadge: View {
    let text: String
    var tone: SettingsTone = .idle

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .foregroundStyle(tone.color)
            .background(tone.color.opacity(0.15), in: Capsule())
            .fixedSize()
    }
}

/// A small coloured dot before a state word.
struct SettingsDot: View {
    let tone: SettingsTone

    var body: some View {
        Circle()
            .fill(tone.color)
            .frame(width: 8, height: 8)
            .accessibilityHidden(true)
    }
}

/// A state word after its dot ("Running", "Missing").
struct SettingsDotText: View {
    let text: String
    let tone: SettingsTone

    var body: some View {
        HStack(spacing: 6) {
            SettingsDot(tone: tone)
            Text(text)
        }
    }
}

/// A section title in the website's case, with something on its right (a badge, mostly).
struct SettingsSectionHeader<Trailing: View>: View {
    let title: String
    @ViewBuilder var trailing: () -> Trailing

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            Text(title)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 8)
            trailing()
        }
        .textCase(nil)
    }
}

/// A large number with its label under it ("0 Waiting").
struct SettingsStat: View {
    let label: String
    let value: Int
    var danger = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value.formatted())
                .font(.title2.weight(.bold).monospacedDigit())
                .foregroundStyle(danger ? Color.red : Color.primary)
            Text(label)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

/// A row that starts work on the server: its label, and a spinner while it runs.
struct SettingsActionRow: View {
    let title: String
    let systemImage: String
    var busy = false
    var role: ButtonRole?
    let action: () -> Void

    var body: some View {
        Button(role: role, action: action) {
            HStack {
                Label(title, systemImage: systemImage)
                Spacer()
                if busy { ProgressView() }
            }
        }
        .disabled(busy)
    }
}
