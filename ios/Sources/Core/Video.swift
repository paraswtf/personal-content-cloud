import AVKit
import SwiftUI

/// Players for the server's videos and Live Photo clips. The token goes as a header, to the server's own host only.
enum AuthorizedPlayer {
    /// An asset for a server address (a "/files/video/..." path from the API), or nil for another host.
    static func asset(for ref: String, vault: Bool = false) -> AVURLAsset? {
        guard let server = APIConnection.current, let url = server.mediaURL(ref, vault: vault) else { return nil }
        return AVURLAsset(url: url, options: ["AVURLAssetHTTPHeaderFieldsKey": server.headers])
    }

    static func player(for ref: String, vault: Bool = false) -> AVPlayer? {
        guard let asset = asset(for: ref, vault: vault) else { return nil }
        return AVPlayer(playerItem: AVPlayerItem(asset: asset))
    }
}

/// A player surface without controls (a Live Photo's clip), filling or fitting its frame.
struct PlayerSurface: UIViewRepresentable {
    let player: AVPlayer
    var fill = false

    func makeUIView(context: Context) -> PlayerLayerView {
        let view = PlayerLayerView()
        view.playerLayer.player = player
        view.playerLayer.videoGravity = fill ? .resizeAspectFill : .resizeAspect
        view.backgroundColor = .clear
        return view
    }

    func updateUIView(_ view: PlayerLayerView, context: Context) {
        if view.playerLayer.player !== player { view.playerLayer.player = player }
        view.playerLayer.videoGravity = fill ? .resizeAspectFill : .resizeAspect
    }

    final class PlayerLayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }
}
