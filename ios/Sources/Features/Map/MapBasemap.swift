import MapKit
import UIKit

/// The plain map (the website's offline basemap, MapView's offlineStyle): Natural Earth land, lakes and borders
/// from basemap.json, a copy of the website's frontend/public/map/basemap.json bundled with the app
/// (backend/tests/test_ios_basemap.py keeps the two the same). It is drawn as MapKit overlays over an empty base,
/// so nothing about the map is asked of anyone but the owner's server. Detailed map is Apple's map, over the internet;
/// the choice is kept on this phone.
enum MapBasemap {
    /// Where the choice is kept: "offline" (the plain map, the default) or "online" (Apple's detailed map), the
    /// website's two values.
    static let storageKey = "map.tiles"
    static let plain = "offline"
    static let detailed = "online"

    /// The basemap's shapes by kind.
    struct Shapes {
        let land: [MKPolygon]
        let lakes: [MKPolygon]
        let borders: [MKPolyline]
    }

    /// Decoded once, the first time the plain map is drawn.
    static let shapes: Shapes? = load()

    /// Reads and decodes basemap.json (land and lake polygons, border lines).
    static func load(from url: URL? = Bundle.main.url(forResource: "basemap", withExtension: "json")) -> Shapes? {
        guard let url, let data = try? Data(contentsOf: url),
              let objects = try? MKGeoJSONDecoder().decode(data) else { return nil }
        var land: [MKPolygon] = []
        var lakes: [MKPolygon] = []
        var borders: [MKPolyline] = []
        for case let feature as MKGeoJSONFeature in objects {
            let kind = kindOf(feature.properties)
            for geometry in feature.geometry {
                switch geometry {
                case let shape as MKMultiPolygon:
                    if kind == "lake" { lakes += shape.polygons } else if kind == "land" { land += shape.polygons }
                case let shape as MKPolygon:
                    if kind == "lake" { lakes.append(shape) } else if kind == "land" { land.append(shape) }
                case let shape as MKMultiPolyline:
                    if kind == "boundary" { borders += shape.polylines }
                case let shape as MKPolyline:
                    if kind == "boundary" { borders.append(shape) }
                default:
                    break
                }
            }
        }
        return Shapes(land: land, lakes: lakes, borders: borders)
    }

    /// A feature's `kind` property (land, lake or boundary).
    static func kindOf(_ properties: Data?) -> String? {
        guard let properties,
              let object = try? JSONSerialization.jsonObject(with: properties) as? [String: Any] else { return nil }
        return object["kind"] as? String
    }

    // The website's colours: water in the soft accent, land in the surface colour, coasts and borders in the line
    // colour, each with its dark variant.
    static let water = UIColor { $0.userInterfaceStyle == .dark
        ? UIColor(red: 0x1e / 255, green: 0x28 / 255, blue: 0x47 / 255, alpha: 1)
        : UIColor(red: 0xe5 / 255, green: 0xea / 255, blue: 0xfb / 255, alpha: 1) }
    static let land = UIColor { $0.userInterfaceStyle == .dark
        ? UIColor(red: 0x1a / 255, green: 0x1d / 255, blue: 0x20 / 255, alpha: 1)
        : UIColor.white }
    static let line = UIColor { $0.userInterfaceStyle == .dark
        ? UIColor(red: 0x30 / 255, green: 0x35 / 255, blue: 0x3a / 255, alpha: 1)
        : UIColor(red: 0xe0 / 255, green: 0xe0 / 255, blue: 0xda / 255, alpha: 1) }
}

/// The plain map's base: one tile of water colour everywhere, replacing Apple's map, so MapKit asks for none of its
/// own tiles.
final class MapPlainTiles: MKTileOverlay {
    private let tile: Data

    init(color: UIColor) {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let side = CGSize(width: 64, height: 64)
        tile = UIGraphicsImageRenderer(size: side, format: format).pngData { context in
            color.setFill()
            context.fill(CGRect(origin: .zero, size: side))
        }
        super.init(urlTemplate: nil)
        canReplaceMapContent = true
    }

    override func loadTile(at path: MKTileOverlayPath, result: @escaping (Data?, Error?) -> Void) {
        result(tile, nil)
    }
}
