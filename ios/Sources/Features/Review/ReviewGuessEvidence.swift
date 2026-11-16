import SwiftUI

/// Why a guess names who it names (the website's components/people/evidence.ts, TraitCompare.tsx and
/// PlaceCompare.tsx): the reasons in words and the crops side by side. The review area builds it for its guess
/// card; the boxes area shows it in its guess chip and face editor. Its name and parameters stay. Placeholder
/// until then: nothing.
struct ReviewGuessEvidence: View {
    let guess: API.GuessOut

    var body: some View {
        EmptyView()
    }
}
