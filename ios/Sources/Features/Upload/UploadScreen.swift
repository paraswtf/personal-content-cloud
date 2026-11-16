import SwiftUI

/// Upload and sync. Placeholder of the core (the upload area replaces this file, keeps these three links and keeps
/// asking for Photos access when it opens): the phone's library sync, removing synced photos from the phone, and the
/// sync's options.
struct UploadScreen: View {
    var body: some View {
        List {
            Section("This phone's photos") {
                NavigationLink { SyncView() } label: { Label("Sync", systemImage: "arrow.triangle.2.circlepath") }
                    .accessibilityIdentifier("upload.sync")
                NavigationLink { DeleteSyncedView() } label: { Label("Delete synced", systemImage: "trash") }
                    .accessibilityIdentifier("upload.deleteSynced")
                NavigationLink { SettingsView() } label: { Label("Sync settings", systemImage: "slider.horizontal.3") }
                    .accessibilityIdentifier("upload.syncSettings")
            }
        }
        .navigationTitle("Upload and sync")
        // Opening upload is where the app first asks for Photos access (never at launch or sign-in).
        .task { await PhotoLibrary.requestAccessIfNeeded() }
    }
}
