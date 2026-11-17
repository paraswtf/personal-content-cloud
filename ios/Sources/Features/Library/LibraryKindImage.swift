import SwiftUI

/// A picture from the server, as the core's RemoteImage, that shows an icon of the item's kind when it cannot be
/// loaded (a document, a video, a file: the website's MediaGrid and trash fall back to the kind's icon). The library's
/// document tiles and the trash use it.
struct LibraryKindImage: View {
    let ref: String?
    var maxPixel = 384
    var contentMode: ContentMode = .fill
    var vault = false
    var kind: MediaKind = .image
    /// Where a fitted picture sits in its frame (a document's page at the top).
    var alignment: Alignment = .center

    @State private var image: UIImage?
    /// The address `image` was loaded from, so a reused view never shows another item's picture.
    @State private var shown: String?
    @State private var failed = false

    init(ref: String?, maxPixel: Int = 384, contentMode: ContentMode = .fill, vault: Bool = false,
         kind: MediaKind = .image, alignment: Alignment = .center) {
        self.ref = ref
        self.maxPixel = maxPixel
        self.contentMode = contentMode
        self.vault = vault
        self.kind = kind
        self.alignment = alignment
        let hit = ref.flatMap { ImageLoader.shared.cached($0, maxPixel: maxPixel, vault: vault) }
        _image = State(initialValue: hit)
        _shown = State(initialValue: hit == nil ? nil : ref)
    }

    /// The icon of an item's kind (the website's kind icons).
    static func symbol(_ kind: MediaKind) -> String {
        switch kind {
        case .image: return "photo"
        case .video: return "film"
        case .pdf, .document: return "doc.text"
        case .other: return "doc"
        }
    }

    var body: some View {
        ZStack {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: contentMode)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: alignment)
            } else {
                Rectangle().fill(Color(uiColor: .secondarySystemFill))
                if failed {
                    Image(systemName: Self.symbol(kind))
                        .font(.title2)
                        .foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                }
            }
        }
        .task(id: "\(ref ?? "")|\(maxPixel)|\(vault)") { await load() }
    }

    private func load() async {
        guard let ref, !ref.isEmpty else {
            image = nil
            failed = true
            return
        }
        if let hit = ImageLoader.shared.cached(ref, maxPixel: maxPixel, vault: vault) {
            image = hit
            shown = ref
            failed = false
            return
        }
        if shown != ref {
            image = nil
            shown = nil
        }
        failed = false
        do {
            let loaded = try await ImageLoader.shared.image(ref, maxPixel: maxPixel, vault: vault)
            image = loaded
            shown = ref
        } catch {
            if !(error is CancellationError) && !Task.isCancelled { failed = true }
        }
    }
}
