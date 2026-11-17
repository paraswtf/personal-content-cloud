import XCTest
@testable import PersonalContentCloud

/// The library's filters as the website sends them (components/library/filters.ts), its search snippets
/// (components/library/Snippet.tsx) and its tile words (components/MediaGrid.tsx describe()).
final class LibraryFilterTests: XCTestCase {
    func testDefaultsAreLeftOut() {
        var f = LibraryFilters()
        f.live = false
        f.favorite = false
        f.misclicks = .exclude
        f.forwards = .exclude
        f.dups = .collapse
        f.sort = .taken
        f.person = []
        f.tag = ["  ", ""]
        f.received = "  "
        XCTAssertEqual(f.normalized(for: .all), LibraryFilters())
    }

    func testSetValuesAreSent() {
        var f = LibraryFilters()
        f.category = .food
        f.kind = .video
        f.person = [4, 4, 0, 9]
        f.live = true
        f.favorite = true
        f.misclicks = .include
        f.forwards = .only
        f.dups = .all
        f.sort = .added
        f.from = "2026-01-02"
        f.to = "2026-03-04"
        let n = f.normalized(for: .album(id: 3))
        XCTAssertEqual(n.category, .food)
        XCTAssertEqual(n.kind, .video)
        XCTAssertEqual(n.person, [4, 9], "people without repeats or bad ids")
        XCTAssertEqual(n.live, true)
        XCTAssertEqual(n.favorite, true)
        XCTAssertEqual(n.misclicks, .include)
        XCTAssertEqual(n.forwards, .only)
        XCTAssertEqual(n.dups, .all)
        XCTAssertEqual(n.sort, .added)
        XCTAssertEqual(n.from, "2026-01-02")
        XCTAssertEqual(n.to, "2026-03-04")
    }

    func testTheVaultHasNoCategoryAndThePrivateCategoryIsNoChip() {
        var f = LibraryFilters()
        f.category = .places
        XCTAssertNil(f.normalized(for: .vault).category)
        XCTAssertEqual(f.normalized(for: .all).category, .places)
        f.category = .nsfw
        XCTAssertNil(f.normalized(for: .all).category)
    }

    func testDatesMustBeDays() {
        var f = LibraryFilters()
        f.from = "2026-1-2"
        f.to = "yesterday"
        XCTAssertNil(f.normalized(for: .all).from)
        XCTAssertNil(f.normalized(for: .all).to)
        let day = LibraryFilters.day("2026-02-14")
        XCTAssertNotNil(day)
        XCTAssertEqual(LibraryFilters.dayString(day!), "2026-02-14")
    }

    func testTagsAreTrimmedUniqueShortAndAtMostTwenty() {
        let long = String(repeating: "x", count: 41)
        XCTAssertEqual(LibraryFilters.cleanTags([" Beach ", "beach", "BEACH", long, "Hills"]), ["Beach", "Hills"])
        let many = (1...30).map { "tag \($0)" }
        XCTAssertEqual(LibraryFilters.cleanTags(many).count, 20)
    }

    func testSearchTakesOnlyWhatTheWebsiteSends() {
        var f = LibraryFilters()
        f.category = .documents
        f.tag = ["Receipts"]
        f.favorite = true
        f.misclicks = .include
        f.forwards = .include
        f.dups = .all
        f.sort = .added
        f.live = true
        let s = f.forSearch(in: .all)
        XCTAssertEqual(s.category, .documents)
        XCTAssertEqual(s.tag, ["Receipts"])
        XCTAssertEqual(s.live, true)
        XCTAssertNil(s.favorite)
        XCTAssertNil(s.misclicks)
        XCTAssertNil(s.forwards)
        XCTAssertNil(s.dups)
        XCTAssertNil(s.sort)
    }

    func testStateFlags() {
        var f = LibraryFilters()
        XCTAssertFalse(f.any)
        XCTAssertFalse(f.hasExtra)
        f.category = .pets
        XCTAssertTrue(f.any)
        XCTAssertTrue(f.narrows)
        XCTAssertFalse(f.hasExtra, "a category alone is a chip, not a filter of the panel")
        f = LibraryFilters()
        f.misclicks = .include
        XCTAssertTrue(f.hasExtra)
        XCTAssertFalse(f.narrows, "including mis-clicks adds items, it does not narrow")
    }

    func testSnippetMarksMatchesAndDecodesEntities() {
        let parts = LibrarySnippet.parts("<b>Packing</b> list &amp; \u{2}tickets\u{3} for &lt;two&gt;")
        XCTAssertEqual(parts, [
            .init(text: "Packing", hit: true),
            .init(text: " list & ", hit: false),
            .init(text: "tickets", hit: true),
            .init(text: " for <two>", hit: false),
        ])
        XCTAssertEqual(LibrarySnippet.plain("a <mark>b</mark> c"), "a b c")
        XCTAssertEqual(LibrarySnippet.decode("&amp;lt;"), "&lt;", "a decoded ampersand is not decoded again")
        XCTAssertEqual(LibrarySnippet.parts("<script>x</script>").map(\.text).joined(), "<script>x</script>",
                       "anything but the markers stays text")
    }

    func testDocumentLabels() {
        XCTAssertEqual(LibraryDocTypes.label("boarding_pass"), "Boarding pass")
        XCTAssertEqual(LibraryDocTypes.label("tax_return"), "Tax return")
        XCTAssertEqual(LibraryDocTypes.label(nil), "Document")
        XCTAssertEqual(LibraryDocTypes.extensionLabel("scan.pdf"), "PDF")
        XCTAssertEqual(LibraryDocTypes.extensionLabel("README"), "FILE")
    }

    func testTileWordsFollowTheWebsite() throws {
        let json = """
        {"uid": "abc", "kind": "image", "version": 1, "filename": "IMG_1.JPG", "width": 10, "height": 10,
         "duration_s": null, "taken_at": "2026-11-10T09:00:00.000Z", "created_at": "2026-11-11T09:00:00.000Z",
         "category": "misc", "favorite": true, "misclick": true, "forward": false, "dup_count": 2, "live": false,
         "urls": {"thumb": "/files/thumb/abc?v=1", "preview": "/files/preview/abc?v=1",
                  "preview_small": "/files/preview-small/abc?v=1", "original": "/files/original/abc?v=1"}}
        """
        let card = try JSONDecoder().decode(API.ItemCard.self, from: Data(json.utf8))
        let label = card.libraryLabel
        XCTAssertTrue(label.hasPrefix("Photo, "), label)
        XCTAssertTrue(label.hasSuffix("favourite, possible mis-click, 2 similar"), label)
    }

    // MARK: chosen people and the scrubber's mark

    func testChosenPersonStates() {
        XCTAssertEqual(LibraryChosenPerson.failed(APIError(status: 404, detail: "Not found")), .gone)
        XCTAssertEqual(LibraryChosenPerson.failed(APIError(status: 500, detail: "boom")), .unknown)
        XCTAssertEqual(LibraryChosenPerson.failed(URLError(.timedOut)), .unknown)
        XCTAssertEqual(LibraryChosenPerson.gone.name(id: 7), "Person not available")
        XCTAssertEqual(LibraryChosenPerson.unknown.name(id: 7), "Person 7")
        let pet = LibraryChosenPerson.known(name: "Biscuit", cover: "/files/face/3", pet: true)
        XCTAssertEqual(pet.name(id: 7), "Biscuit")
        XCTAssertTrue(pet.isPet)
        XCTAssertEqual(pet.cover, "/files/face/3")
        XCTAssertFalse(LibraryChosenPerson.gone.isPet)
        XCTAssertTrue(LibraryChosenPerson.gone.isGone)
    }

    func testScrollFraction() {
        XCTAssertEqual(LibraryScrollPosition.fraction(contentTop: 120, contentHeight: 5000, viewport: 700), 0)
        XCTAssertEqual(LibraryScrollPosition.fraction(contentTop: -2150, contentHeight: 5000, viewport: 700), 0.5,
                       accuracy: 0.0001)
        XCTAssertEqual(LibraryScrollPosition.fraction(contentTop: -9000, contentHeight: 5000, viewport: 700), 1)
        // shorter than the screen: nothing to scroll
        XCTAssertEqual(LibraryScrollPosition.fraction(contentTop: -10, contentHeight: 300, viewport: 700), 0)
    }
}
