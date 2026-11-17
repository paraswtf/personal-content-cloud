import SwiftUI

/// The Library tab (the website's pages/LibraryPage.tsx): the whole library in the library's browser.
struct LibraryScreen: View {
    var body: some View {
        LibraryBrowser(scope: .all)
            .navigationTitle("Library")
    }
}
