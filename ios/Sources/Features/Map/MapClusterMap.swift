import SwiftUI
import MapKit

/// The map itself (docs/ARCHITECTURE.md section 25, the website's MapView): the plain basemap drawn from the app's own
/// copy of the website's basemap (MapBasemap) or, with `detailed`, Apple's map, with every point as an annotation that
/// MapKit clusters, clusters drawn with their counts, dots for single points and their thumbnails
/// once zoomed in, and the taps: a big cluster zooms in, a small cluster or a single point reports a box for the
/// panel. No location permission: the map never shows where the phone is.
struct MapClusterMap: UIViewRepresentable {
    let points: [API.MapPointsPointsTuple]
    let flyTo: MapFlyTarget?
    /// Apple's map over the internet instead of the plain basemap.
    var detailed = false
    let onPick: (MapPickedArea) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> MKMapView {
        let map = MKMapView(frame: .zero)
        map.delegate = context.coordinator
        map.showsUserLocation = false
        map.isRotateEnabled = false
        map.isPitchEnabled = false
        map.pointOfInterestFilter = .excludingAll
        map.showsCompass = false
        map.register(MapPointView.self, forAnnotationViewWithReuseIdentifier: MapPointView.reuseId)
        map.register(MapClusterView.self, forAnnotationViewWithReuseIdentifier: MapClusterView.reuseId)
        map.accessibilityLabel = "Map of your photos"
        return map
    }

    func updateUIView(_ map: MKMapView, context: Context) {
        context.coordinator.parent = self
        context.coordinator.applyBasemap(detailed: detailed, dark: context.environment.colorScheme == .dark, to: map)
        context.coordinator.apply(points, to: map)
        context.coordinator.fly(flyTo, on: map)
    }

    @MainActor
    final class Coordinator: NSObject, MKMapViewDelegate {
        var parent: MapClusterMap
        private var signature = ""
        private var fitted = false
        private var lastNonce: Int?
        private var showThumbs = false
        /// The basemap on the map now ("online", or the plain map in light or dark), so an update that changes
        /// nothing leaves the overlays alone.
        private var basemapKey: String?
        private var basemapOverlays: [MKOverlay] = []
        private var lakes: MKMultiPolygon?
        private var colors = (water: UIColor.clear, land: UIColor.clear, line: UIColor.clear)

        init(_ parent: MapClusterMap) {
            self.parent = parent
        }

        /// The plain map: a base of water replacing Apple's map, then the land with its coast, the lakes and the
        /// borders, in the website's colours for light or dark. Detailed: Apple's map, no overlays.
        func applyBasemap(detailed: Bool, dark: Bool, to map: MKMapView) {
            let key = detailed ? MapBasemap.detailed : "\(MapBasemap.plain)-\(dark ? "dark" : "light")"
            guard key != basemapKey else { return }
            basemapKey = key
            map.removeOverlays(basemapOverlays)
            basemapOverlays = []
            lakes = nil
            guard !detailed else { return }
            let traits = UITraitCollection(userInterfaceStyle: dark ? .dark : .light)
            colors = (water: MapBasemap.water.resolvedColor(with: traits),
                      land: MapBasemap.land.resolvedColor(with: traits),
                      line: MapBasemap.line.resolvedColor(with: traits))
            var overlays: [MKOverlay] = [MapPlainTiles(color: colors.water)]
            if let shapes = MapBasemap.shapes {
                let lakeShape = MKMultiPolygon(shapes.lakes)
                lakes = lakeShape
                overlays += [MKMultiPolygon(shapes.land), lakeShape, MKMultiPolyline(shapes.borders)] as [MKOverlay]
            }
            // added in order at one level: the base first, the borders last, on top
            map.addOverlays(overlays, level: .aboveLabels)
            basemapOverlays = overlays
        }

        func mapView(_ mapView: MKMapView, rendererFor overlay: MKOverlay) -> MKOverlayRenderer {
            if let tiles = overlay as? MKTileOverlay {
                return MKTileOverlayRenderer(tileOverlay: tiles)
            }
            if let area = overlay as? MKMultiPolygon {
                let renderer = MKMultiPolygonRenderer(multiPolygon: area)
                if area === lakes {
                    renderer.fillColor = colors.water
                } else {
                    renderer.fillColor = colors.land
                    renderer.strokeColor = colors.line
                    renderer.lineWidth = 0.8
                }
                return renderer
            }
            if let lines = overlay as? MKMultiPolyline {
                let renderer = MKMultiPolylineRenderer(multiPolyline: lines)
                renderer.strokeColor = colors.line
                renderer.lineWidth = 1
                return renderer
            }
            return MKOverlayRenderer(overlay: overlay)
        }

        /// New points replace the annotations; the same points again (a refresh) leave the clusters alone.
        func apply(_ points: [API.MapPointsPointsTuple], to map: MKMapView) {
            var lat = 0.0, lon = 0.0
            for p in points { lat += p.v0; lon += p.v1 }
            let key = "\(points.count):\(String(format: "%.4f", lat)):\(String(format: "%.4f", lon)):"
                + "\(points.first?.v2 ?? ""):\(points.last?.v2 ?? "")"
            guard key != signature else { return }
            signature = key
            map.removeAnnotations(map.annotations)
            let annotations = points.map(MapPointAnnotation.init)
            map.addAnnotations(annotations)
            if !fitted, !annotations.isEmpty {
                fitted = true
                if map.bounds.width > 0 {
                    fit(map, annotations)
                } else {
                    DispatchQueue.main.async { [weak self, weak map] in
                        guard let self, let map else { return }
                        self.fit(map, annotations)
                    }
                }
            }
        }

        /// The first view shows every point, no closer than about zoom 11 (the website's fitBounds).
        private func fit(_ map: MKMapView, _ annotations: [MapPointAnnotation]) {
            var rect = MKMapRect.null
            for a in annotations {
                let p = MKMapPoint(a.coordinate)
                rect = rect.union(MKMapRect(x: p.x, y: p.y, width: 0, height: 0))
            }
            var region = MKCoordinateRegion(rect)
            region.span.latitudeDelta = max(region.span.latitudeDelta * 1.3, MapRules.closestSpan)
            region.span.longitudeDelta = max(region.span.longitudeDelta * 1.3, MapRules.closestSpan)
            map.setRegion(map.regionThatFits(region), animated: false)
        }

        /// Flies to a place from the list, or to the spot the map was opened at (Show on map).
        func fly(_ target: MapFlyTarget?, on map: MKMapView) {
            guard let target, target.nonce != lastNonce else { return }
            lastNonce = target.nonce
            fitted = true
            let region = MKCoordinateRegion(
                center: CLLocationCoordinate2D(latitude: target.latitude, longitude: target.longitude),
                span: MKCoordinateSpan(latitudeDelta: target.span, longitudeDelta: target.span))
            if map.bounds.width > 0 {
                map.setRegion(region, animated: true)
            } else {
                // Opened at a spot: the first update comes before the map has a size, and a region set on a map
                // without one does not keep its zoom. Set it once the layout has given it one, as the first fit does.
                DispatchQueue.main.async { [weak map] in
                    guard let map else { return }
                    map.setRegion(map.bounds.width > 0 ? map.regionThatFits(region) : region, animated: false)
                }
            }
        }

        func mapView(_ mapView: MKMapView, viewFor annotation: MKAnnotation) -> MKAnnotationView? {
            if let cluster = annotation as? MKClusterAnnotation {
                let view = mapView.dequeueReusableAnnotationView(withIdentifier: MapClusterView.reuseId, for: cluster)
                (view as? MapClusterView)?.configure(cluster)
                return view
            }
            if let point = annotation as? MapPointAnnotation {
                let view = mapView.dequeueReusableAnnotationView(withIdentifier: MapPointView.reuseId, for: point)
                (view as? MapPointView)?.configure(point, thumb: showThumbs)
                return view
            }
            return nil
        }

        func mapView(_ mapView: MKMapView, didSelect view: MKAnnotationView) {
            guard let annotation = view.annotation else { return }
            mapView.deselectAnnotation(annotation, animated: false)
            if let cluster = annotation as? MKClusterAnnotation {
                let members = cluster.memberAnnotations
                let coordinates = members.map(\.coordinate)
                if MapRules.zoomsIn(count: members.count, coordinates: coordinates) {
                    mapView.showAnnotations(members, animated: true)
                    return
                }
                if let area = MapPickedArea.around(coordinates, count: members.count, pad: MapRules.clusterPad) {
                    parent.onPick(area)
                }
                return
            }
            if let area = MapPickedArea.around([annotation.coordinate], count: 1, pad: MapRules.pointPad) {
                parent.onPick(area)
            }
        }

        /// Thumbnails replace the dots once zoomed in, and the dots come back when zoomed out.
        func mapView(_ mapView: MKMapView, regionDidChangeAnimated animated: Bool) {
            let thumbs = mapView.region.span.longitudeDelta < MapRules.thumbSpan
            guard thumbs != showThumbs else { return }
            showThumbs = thumbs
            for annotation in mapView.annotations {
                guard let point = annotation as? MapPointAnnotation,
                      let view = mapView.view(for: point) as? MapPointView else { continue }
                view.configure(point, thumb: thumbs)
            }
        }
    }
}

/// A single point: a dot, or its thumbnail once zoomed in.
final class MapPointView: MKAnnotationView {
    static let reuseId = "map.point"
    static let clusterId = "map.points"

    private let thumbView = UIImageView()
    private var loading: Task<Void, Never>?
    private var shownUid: String?

    override init(annotation: MKAnnotation?, reuseIdentifier: String?) {
        super.init(annotation: annotation, reuseIdentifier: reuseIdentifier)
        clusteringIdentifier = Self.clusterId
        collisionMode = .circle
        displayPriority = .defaultHigh
        thumbView.contentMode = .scaleAspectFill
        thumbView.clipsToBounds = true
        addSubview(thumbView)
        accessibilityTraits = .button
        accessibilityIdentifier = "map.point"
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.25
        layer.shadowRadius = 3
        layer.shadowOffset = CGSize(width: 0, height: 1)
    }

    required init?(coder: NSCoder) { nil }

    /// MapKit keeps the views of points inside a cluster but does not show them; VoiceOver skips them too.
    override var isAccessibilityElement: Bool {
        get { !isHidden && alpha > 0.01 }
        set {}
    }

    override func prepareForReuse() {
        super.prepareForReuse()
        loading?.cancel()
        loading = nil
        thumbView.image = nil
        shownUid = nil
    }

    func configure(_ point: MapPointAnnotation, thumb: Bool) {
        clusteringIdentifier = Self.clusterId
        if thumb {
            frame = CGRect(x: 0, y: 0, width: 40, height: 40)
            layer.cornerRadius = 8
            layer.borderWidth = 2
            layer.borderColor = UIColor.systemBackground.cgColor
            backgroundColor = .secondarySystemBackground
            thumbView.frame = bounds.insetBy(dx: 2, dy: 2)
            thumbView.layer.cornerRadius = 6
            thumbView.isHidden = false
            accessibilityLabel = "Open this photo"
            load(point.uid)
        } else {
            loading?.cancel()
            loading = nil
            frame = CGRect(x: 0, y: 0, width: 14, height: 14)
            layer.cornerRadius = 7
            layer.borderWidth = 2
            layer.borderColor = UIColor.white.cgColor
            backgroundColor = tintColor
            thumbView.isHidden = true
            accessibilityLabel = "1 item here"
        }
    }

    private func load(_ uid: String) {
        guard shownUid != uid || thumbView.image == nil else { return }
        shownUid = uid
        let ref = "/files/thumb/\(uid)"
        if let cached = ImageLoader.shared.cached(ref, maxPixel: 160) {
            thumbView.image = cached
            return
        }
        thumbView.image = nil
        loading?.cancel()
        loading = Task { [weak self] in
            let image = try? await ImageLoader.shared.image(ref, maxPixel: 160)
            guard !Task.isCancelled, let self, self.shownUid == uid else { return }
            self.thumbView.image = image
        }
    }
}

/// A cluster: a round marker with how many items are under it.
final class MapClusterView: MKAnnotationView {
    static let reuseId = "map.cluster"

    private let label = UILabel()

    override init(annotation: MKAnnotation?, reuseIdentifier: String?) {
        super.init(annotation: annotation, reuseIdentifier: reuseIdentifier)
        collisionMode = .circle
        displayPriority = .defaultHigh
        label.textAlignment = .center
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .semibold)
        label.adjustsFontSizeToFitWidth = true
        label.minimumScaleFactor = 0.7
        addSubview(label)
        isAccessibilityElement = true
        accessibilityTraits = .button
        accessibilityIdentifier = "map.cluster"
        layer.borderWidth = 2
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.25
        layer.shadowRadius = 3
        layer.shadowOffset = CGSize(width: 0, height: 1)
    }

    required init?(coder: NSCoder) { nil }

    override var annotation: MKAnnotation? {
        didSet {
            if let cluster = annotation as? MKClusterAnnotation { configure(cluster) }
        }
    }

    func configure(_ cluster: MKClusterAnnotation) {
        let count = cluster.memberAnnotations.count
        let side = MapRules.clusterSize(count)
        frame = CGRect(x: 0, y: 0, width: side, height: side)
        layer.cornerRadius = side / 2
        layer.borderColor = UIColor.systemBackground.cgColor
        backgroundColor = tintColor
        label.frame = bounds.insetBy(dx: 3, dy: 3)
        label.text = MapRules.clusterText(count)
        accessibilityLabel = "\(count.formatted()) items here"
    }
}
