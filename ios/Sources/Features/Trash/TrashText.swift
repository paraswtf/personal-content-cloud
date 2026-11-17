import Foundation

/// The trash's words, as the website's TrashPage says them, apart from the screens so they can be tested.
enum TrashText {
    static func intro(days: Int) -> String {
        "Deleted items stay here for \(days) days, then they and their files are removed for good. Restore puts an "
            + "item back exactly as it was."
    }

    static let emptyTitle = "The trash is empty"

    static func emptyText(days: Int) -> String {
        "Items you delete appear here for \(days) days before they are removed for good."
    }

    /// "1 item", "12 items".
    static func count(_ n: Int) -> String {
        "\(n.formatted()) \(n == 1 ? "item" : "items")"
    }

    /// After Restore: "IMG_1.JPG is back in the library."
    static func restored(_ filename: String?) -> String {
        guard let filename, !filename.isEmpty else { return "Restored." }
        return "\(filename) is back in the library."
    }

    static let purged = "Deleted for good."

    /// After Empty trash: "1 item deleted for good.", "3 items deleted for good."
    static func emptied(_ n: Int) -> String {
        n == 1 ? "1 item deleted for good." : "\(n.formatted()) items deleted for good."
    }

    /// The question before Delete forever.
    static func purgeQuestion(_ filename: String) -> String {
        "\(filename) and all its files will be deleted for good. This cannot be undone."
    }

    /// The question before Empty trash.
    static func emptyQuestion(_ total: Int) -> String {
        "All \(count(total)) in the trash and their files will be deleted for good. This cannot be undone."
    }

    /// "Video, 1:15" under a video in the larger look (the website's TrashPreview); nil for anything else.
    static func videoLength(kind: String, seconds: Double?) -> String? {
        guard kind == "video", let seconds, seconds > 0 else { return nil }
        return "Video, \(Fmt.duration(seconds))"
    }

    /// "2 of 14" in the larger look.
    static func position(_ index: Int, of total: Int) -> String {
        "\((index + 1).formatted()) of \(total.formatted())"
    }
}
