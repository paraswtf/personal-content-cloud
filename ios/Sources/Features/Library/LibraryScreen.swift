import SwiftUI

/// The Library tab. Placeholder until the library area builds it: the whole library in the library's browser.
struct LibraryScreen: View {
    var body: some View {
        LibraryBrowser(scope: .all)
            .navigationTitle("Library")
    }
}
