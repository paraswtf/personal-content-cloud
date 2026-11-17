import SwiftUI

/// The viewer's buttons in the bottom bar, between the core's boxes button and Details (the website's header:
/// favourite, share and download, with the info panel's actions): favourite, share (not for vault items), and a menu
/// with the downloads (the original, the Live Photo clip, the video's frame on screen, the document's page), saving to
/// Photos, adding to an album, the vault, processing again and moving to the trash. The boxes area's own buttons
/// follow.
struct ViewerToolbar: View {
    let context: ViewerPageContext
    let boxes: ViewerBoxesModel
    let pages: ViewerPageState

    @Environment(AppRouter.self) private var router
    @State private var sharing = false
    @State private var addingToAlbum = false
    @State private var confirmTrash = false
    @State private var trashDays = 30
    @State private var file: ViewerFile?
    @State private var working = false

    var body: some View {
        HStack(spacing: 2) {
            if let item = context.item {
                Button {
                    Task { await ViewerActions.setFavourite(context, !item.favorite) }
                } label: {
                    Image(systemName: item.favorite ? "star.fill" : "star")
                        .foregroundStyle(item.favorite ? Color.yellow : Color.white)
                        .frame(width: 44, height: 44)
                }
                .accessibilityLabel(item.favorite ? "Remove from favourites" : "Add to favourites")
                .accessibilityIdentifier("viewer.favourite")
                if !item.nsfw && !context.vault {
                    Button {
                        sharing = true
                    } label: {
                        Image(systemName: "square.and.arrow.up").frame(width: 44, height: 44)
                    }
                    .accessibilityLabel("Share")
                    .accessibilityIdentifier("viewer.share")
                }
                menu(item)
            }
            ViewerBoxesToolbarItems(model: boxes, context: context)
        }
        .sheet(isPresented: $sharing) { SharingCreateSheet(target: .items(uids: [context.uid])) }
        .sheet(isPresented: $addingToAlbum) {
            AlbumsAddSheet(uids: [context.uid]) {
                context.reload()
                CloudEvents.post(items: [context.uid])
            }
        }
        .sheet(item: $file) { file in
            ViewerShareSheet(items: [file.url]).onDisappear { ViewerFiles.discard(file) }
        }
        .confirmAction("Move this \(ViewerFormat.trashNoun(MediaKind(context.item?.kind.rawValue))) to the trash?",
                       isPresented: $confirmTrash, actionTitle: "Move to trash",
                       message: "It leaves the library now and is deleted for good after \(trashDays) days. Until then you can restore it from Trash with its tags and document details.") {
            Task { await ViewerActions.trash(context, router: router) }
        }
        .task { trashDays = await ViewerConfig.loadTrashDays() }
    }

    private func menu(_ item: API.ItemDetail) -> some View {
        let kind = MediaKind(item.kind.rawValue)
        return Menu {
            if let download = item.urls.download {
                Section("Download") {
                    Button {
                        Task { await share(address: download, filename: item.filename) }
                    } label: {
                        Label("Download original, \(Fmt.bytes(item.sizeBytes))", systemImage: "arrow.down.circle")
                    }
                    if item.live, let clip = item.urls.live ?? item.liveVideo?.url {
                        Button {
                            Task { await share(address: clip, filename: Self.clipName(item.filename)) }
                        } label: {
                            Label("Download the Live Photo clip", systemImage: "livephoto")
                        }
                    }
                    if kind == .video {
                        Button {
                            Task { await shareFrame(item) }
                        } label: {
                            Label("Download this frame", systemImage: "photo")
                        }
                    }
                    if (kind == .pdf || kind == .document), !item.urls.pages.isEmpty {
                        let index = min(max(pages.page[item.uid] ?? 0, 0), item.urls.pages.count - 1)
                        Button {
                            Task { await share(address: item.urls.pages[index], filename: Self.pageName(item.filename, index + 1)) }
                        } label: {
                            Label("Download this page as an image", systemImage: "doc.richtext")
                        }
                    }
                    // Not for private items: the phone's library is outside the vault (and may sync to iCloud).
                    if (kind == .image || kind == .video) && !item.nsfw && !context.vault {
                        Button {
                            Task { await saveToPhotos(address: download, filename: item.filename) }
                        } label: {
                            Label("Save to Photos", systemImage: "photo.badge.plus")
                        }
                    }
                }
            }
            Section {
                Button {
                    addingToAlbum = true
                } label: {
                    Label("Add to album", systemImage: "rectangle.stack.badge.plus")
                }
                Button {
                    Task { await ViewerActions.setPrivate(context, !item.nsfw, router: router) }
                } label: {
                    Label(item.nsfw ? "Not private" : "Mark private", systemImage: item.nsfw ? "lock.open" : "lock")
                }
            }
            Section {
                Button {
                    Task { await ViewerActions.reprocess(context, .analyze) }
                } label: {
                    Label("Analyse again", systemImage: "arrow.clockwise")
                }
                .disabled(item.analysisStatus == .pending)
                if kind == .pdf || kind == .document || item.textStatus != .none {
                    Button {
                        Task { await ViewerActions.reprocess(context, .text) }
                    } label: {
                        Label("Read text again", systemImage: "text.viewfinder")
                    }
                    .disabled(item.textStatus == .pending)
                }
                if kind == .video {
                    Button {
                        Task { await ViewerActions.reprocess(context, .preview) }
                    } label: {
                        Label("Remake video copy", systemImage: "film")
                    }
                    .disabled(item.previewStatus == .pending)
                }
            }
            Button(role: .destructive) {
                confirmTrash = true
            } label: {
                Label("Move to trash", systemImage: "trash")
            }
        } label: {
            Group {
                if working {
                    ProgressView().tint(.white)
                } else {
                    Image(systemName: "ellipsis.circle")
                }
            }
            .frame(width: 44, height: 44)
        }
        .accessibilityLabel("More")
        .accessibilityIdentifier("viewer.more")
    }

    /// "IMG_0001.HEIC" -> "IMG_0001.MOV": the clip of a Live Photo, under the photo's name.
    static func clipName(_ filename: String) -> String {
        let base = (filename as NSString).deletingPathExtension
        return (base.isEmpty ? "Live Photo" : base) + ".MOV"
    }

    /// "Ticket.pdf", page 2 -> "Ticket page 2.webp" (the server renders pages as WebP).
    static func pageName(_ filename: String, _ page: Int) -> String {
        let base = (filename as NSString).deletingPathExtension
        return "\(base.isEmpty ? "Page" : base) page \(page).webp"
    }

    /// The frame on screen as a full-size JPEG (the website's "Download this frame").
    private func shareFrame(_ item: API.ItemDetail) async {
        working = true
        defer { working = false }
        let t = pages.frameTime(item.uid)
        do {
            let request = try APIContext.$vault.withValue(context.vault) {
                try Cloud.api().frameRequest(uid: item.uid, t: (t * 1000).rounded() / 1000,
                                             v: ViewerActions.fileVersion(item), size: "full", download: 1)
            }
            let base = (item.filename as NSString).deletingPathExtension
            file = try await ViewerFiles.fetch(context, request: request,
                                               filename: "\(base) at \(Fmt.duration(t).replacingOccurrences(of: ":", with: "."))).jpg")
        } catch {
            Toaster.shared.error(error, doing: "download the frame")
        }
    }

    private func share(address: String, filename: String) async {
        working = true
        defer { working = false }
        do {
            file = try await ViewerFiles.fetch(context, address: address, filename: filename)
        } catch {
            Toaster.shared.error(error, doing: "download it")
        }
    }

    private func saveToPhotos(address: String, filename: String) async {
        working = true
        defer { working = false }
        do {
            let fetched = try await ViewerFiles.fetch(context, address: address, filename: filename)
            defer { ViewerFiles.discard(fetched) }
            try await ViewerFiles.saveToPhotos(fetched)
            Toaster.shared.ok("Saved to Photos")
        } catch {
            Toaster.shared.error(error, doing: "save it to Photos")
        }
    }
}
