import SwiftUI

/// One guess to confirm, correct or reject (the website's GuessCard): the face or person box, whose it might be and
/// why, and the buttons. The review area builds it; the person page's Guesses tab shows it too, so its name and
/// signature stay. `onDecided` runs once the owner answered it, so the list can take the card away.
/// Placeholder until then.
struct ReviewGuessCard: View {
    let guess: API.GuessItem
    var onDecided: (() -> Void)? = nil

    var body: some View {
        Label("A guess to review", systemImage: "questionmark.circle")
            .foregroundStyle(.secondary)
    }
}
