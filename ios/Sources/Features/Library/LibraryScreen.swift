import SwiftUI

/// The Library tab (the website's pages/LibraryPage.tsx): the whole library in the library's browser.
struct LibraryScreen: View {
    var body: some View {
        // the browser sets the title: "Search results" while searching, as the website's heading
        LibraryBrowser(scope: .all, title: "Library")
    }
}
