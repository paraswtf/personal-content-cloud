import SwiftUI

// The system side of Settings (components/shell/settings: SystemTab with StorageCards, ThroughputCard and
// VideoCard, LearningTab; components/admin; components/undo/UndoHistory). Settings opens these three screens by
// these names. Placeholders until the area builds them.

struct SettingsSystemScreen: View {
    var body: some View { PlaceholderScreen(title: "System", systemImage: "server.rack") }
}

struct SettingsLearningScreen: View {
    var body: some View { PlaceholderScreen(title: "Learning", systemImage: "brain") }
}

struct SettingsUndoHistoryScreen: View {
    var body: some View { PlaceholderScreen(title: "Undo history", systemImage: "arrow.uturn.backward.circle") }
}
