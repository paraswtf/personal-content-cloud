import Foundation
import MapKit

/// A spot on the map the owner chose: the items inside `bbox` (west, south, east, north) go in the panel.
/// `count` is what the marker said, shown until the list's own total arrives.
struct MapPickedArea: Hashable {
    let west: Double
    let south: Double
    let east: Double
    let north: Double
    let count: Int

    /// "w,s,e,n" for the items endpoint, five decimals (the website's bboxParam).
    var bboxParam: String {
        func f(_ v: Double) -> String {
            let rounded = (v * 1e5).rounded() / 1e5
            return rounded == rounded.rounded() ? String(Int(rounded)) : String(rounded)
        }
        return "\(f(west)),\(f(south)),\(f(east)),\(f(north))"
    }

    /// The box around some coordinates, padded by `pad` degrees on every side.
    static func around(_ coordinates: [CLLocationCoordinate2D], count: Int, pad: Double) -> MapPickedArea? {
        guard let first = coordinates.first else { return nil }
        var w = first.longitude, e = first.longitude, s = first.latitude, n = first.latitude
        for c in coordinates {
            w = min(w, c.longitude); e = max(e, c.longitude)
            s = min(s, c.latitude); n = max(n, c.latitude)
        }
        return MapPickedArea(west: w - pad, south: s - pad, east: e + pad, north: n + pad, count: count)
    }
}

/// Where the map should fly: a place from the list, or an item's spot. `nonce` makes a second pick of the same place
/// fly again; `span` is how much of the map shows (degrees).
struct MapFlyTarget: Equatable {
    let latitude: Double
    let longitude: Double
    let nonce: Int
    var span: Double = MapRules.closestSpan
}

/// One photo, video or document with a location (`MapPoints.points`: latitude, longitude, uid, kind code
/// 0 image, 1 video, 2 document, 3 other).
final class MapPointAnnotation: NSObject, MKAnnotation {
    let uid: String
    let kindCode: Int
    let coordinate: CLLocationCoordinate2D

    init(uid: String, kindCode: Int, coordinate: CLLocationCoordinate2D) {
        self.uid = uid
        self.kindCode = kindCode
        self.coordinate = coordinate
    }

    convenience init(_ point: API.MapPointsPointsTuple) {
        self.init(uid: point.v2, kindCode: point.v3,
                  coordinate: CLLocationCoordinate2D(latitude: point.v0, longitude: point.v1))
    }
}

/// The map's rules, shared by the view and the tests (the website's MapView constants).
enum MapRules {
    /// Clusters this small open the panel instead of zooming in.
    static let openBelow = 12
    /// A cluster whose photos sit closer together than this (degrees) cannot split by zooming: it opens the panel.
    static let tightSpan = 0.0015
    /// Single points show their thumbnail once the visible width is under this many degrees (about zoom 12).
    static let thumbSpan = 0.35
    /// The narrowest view the first fit or a chosen place zooms to (about zoom 11 on a phone).
    static let closestSpan = 0.27
    /// How close an item's spot opens (the viewer's Show on map, the website's z=14).
    static let spotSpan = 0.03
    /// Padding of a selection around a cluster's photos and around a single point (degrees).
    static let clusterPad = 0.0002
    static let pointPad = 0.0001

    /// A cluster marker's text: "7", "1.2k" (the website's clusterElement).
    static func clusterText(_ count: Int) -> String {
        guard count >= 1000 else { return String(count) }
        let thousands = (Double(count) / 100).rounded() / 10
        return thousands == thousands.rounded() ? "\(Int(thousands))k" : "\(thousands)k"
    }

    /// A cluster marker's side in points (the website's sizes).
    static func clusterSize(_ count: Int) -> CGFloat {
        count >= 1000 ? 52 : count >= 100 ? 44 : count >= 10 ? 36 : 30
    }

    /// Whether a tapped cluster zooms in (true) or opens the panel (false).
    static func zoomsIn(count: Int, coordinates: [CLLocationCoordinate2D]) -> Bool {
        guard count > openBelow, let first = coordinates.first else { return false }
        var w = first.longitude, e = first.longitude, s = first.latitude, n = first.latitude
        for c in coordinates {
            w = min(w, c.longitude); e = max(e, c.longitude)
            s = min(s, c.latitude); n = max(n, c.latitude)
        }
        return max(e - w, n - s) >= tightSpan
    }

    /// The header's line: "27 items of 41 have a location." (the website's countLine).
    static func countLine(_ points: API.MapPoints?) -> String {
        guard let points else { return "Loading" }
        if points.total == 0 { return "No items match these filters." }
        return "\(Fmt.count(points.withGps, "item")) of \(points.total.formatted()) "
            + "\(points.total == 1 ? "has" : "have") a location."
    }
}
