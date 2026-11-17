import SwiftUI

// The system side of Settings (components/shell/settings: SystemTab with StorageCards, ThroughputCard and
// VideoCard, LearningTab; components/undo/UndoHistory). Settings opens these three screens by these names.

struct SettingsSystemScreen: View {
    var body: some View { SettingsSystemView() }
}

struct SettingsLearningScreen: View {
    var body: some View { SettingsLearningView() }
}

struct SettingsUndoHistoryScreen: View {
    var body: some View { SettingsUndoHistoryView() }
}
