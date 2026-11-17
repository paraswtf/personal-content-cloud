import SwiftUI

/// What an area shows until its screens are built (each Features/<Area>/<Area>Screen.swift replaces it).
struct PlaceholderScreen: View {
    let title: String
    let systemImage: String
    var text = "This part of the app is on its way. The website has it today."

    var body: some View {
        ContentUnavailableView {
            Label(title, systemImage: systemImage)
        } description: {
            Text(text)
        }
        .navigationTitle(title)
    }
}
