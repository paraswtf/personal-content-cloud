import SwiftUI

/// The map (docs/ARCHITECTURE.md section 25, the website's MapPage): the photos with a location, clustered, with the
/// library's filters, the places in a sheet, a panel with the items under a cluster or point, the viewer over it,
/// a status line while older items are still being checked for a location, and the Detailed map choice (the plain
/// basemap by default, Apple's map when chosen, kept on this phone).
struct MapScreen: View {
    @AppStorage(MapBasemap.storageKey) private var tiles = MapBasemap.plain
    @State private var filters = LibraryFilters()
    @State private var points: API.MapPoints?
    @State private var pointsError: Error?
    @State private var places: [API.PlaceOut]?
    @State private var status: API.MapStatus?
    @State private var picked: MapPickedArea?
    @State private var flyTo: MapFlyTarget?
    @State private var activePlace: String?
    @State private var placesOpen = false
    @State private var scanning = false
    @State private var flyCount = 0

    /// `start`: the library's filters, or an item's spot to fly to and list (the website's /map?at=).
    init(start: MapStart? = nil) {
        _filters = State(initialValue: start?.filters ?? LibraryFilters())
        if let lat = start?.latitude, let lon = start?.longitude {
            let pad = 0.0005
            _flyCount = State(initialValue: 1)
            // The spot's panel covers the lower half of the map: the map centres a little south of the spot, so the
            // spot itself shows above the panel. As close as the website's z=14.
            _flyTo = State(initialValue: MapFlyTarget(latitude: lat - MapRules.spotSpan * 0.28, longitude: lon,
                                                      nonce: 1, span: MapRules.spotSpan))
            _picked = State(initialValue: MapPickedArea(west: lon - pad, south: lat - pad, east: lon + pad,
                                                        north: lat + pad, count: 1))
        }
    }

    var body: some View {
        VStack(spacing: 8) {
            header
            LibraryFilterBar(filters: $filters, scope: .all)
            GeometryReader { geo in
                ZStack(alignment: .bottom) {
                    mapLayer
                    if let picked {
                        MapItemsPanel(area: picked, filters: filters) { self.picked = nil }
                            .frame(height: max(220, geo.size.height * 0.55))
                            .id(picked)
                            .transition(.move(edge: .bottom))
                    }
                }
                .animation(.easeInOut(duration: 0.2), value: picked)
            }
        }
        .navigationTitle("Map")
        .navigationBarTitleDisplayMode(.inline)
        // the tab bar keeps its background over the map instead of turning see-through
        .toolbarBackground(.visible, for: .tabBar)
        .sheet(isPresented: $placesOpen) {
            MapPlacesSheet(places: places, active: activePlace) { place in pick(place) }
        }
        .task(id: filters) { await load() }
        .task(id: status?.running == true) { await followScan() }
        .onChange(of: filters) { _, _ in picked = nil }
        .onCloudChange { Task { await load() } }
    }

    // MARK: header

    private var detailed: Bool { tiles == MapBasemap.detailed }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(MapRules.countLine(points))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("map.count")
                Button {
                    tiles = detailed ? MapBasemap.plain : MapBasemap.detailed
                } label: {
                    Label("Detailed map", systemImage: "globe")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 12)
                        .frame(height: 34)
                        .foregroundStyle(detailed ? Color(uiColor: .systemBackground) : Color.primary)
                        .background(detailed ? Color.primary : Color(uiColor: .secondarySystemFill), in: Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(detailed ? .isSelected : [])
                .accessibilityIdentifier("map.detailed")
            }
            Text(detailed
                 ? "The detailed map comes from Apple Maps over the internet; your photos never leave this server."
                 : "The plain map works offline. The detailed map needs the internet.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("map.tilesNote")
            if let status, status.unchecked > 0 || !status.gazetteer || status.running {
                MapStatusBanner(status: status, scanning: scanning) { Task { await scan() } }
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 4)
    }

    // MARK: map

    @ViewBuilder
    private var mapLayer: some View {
        if let pointsError, points == nil {
            ErrorState(error: pointsError) { Task { await load() } }
        } else {
            // The places button sits at the top: at the bottom it would cover the map's Apple Maps and Legal marks.
            ZStack(alignment: .topLeading) {
                MapClusterMap(points: points?.points ?? [], flyTo: flyTo, detailed: detailed) { area in picked = area }
                if let points, points.withGps == 0 {
                    Text(points.total == 0 ? "Nothing matches these filters." : "None of these items has a location yet.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .allowsHitTesting(false)
                }
                Button {
                    placesOpen = true
                } label: {
                    Label(places.map { "Places (\($0.count))" } ?? "Places", systemImage: "mappin.and.ellipse")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 14)
                        .frame(height: 40)
                        .background(.regularMaterial, in: Capsule())
                        .shadow(color: .black.opacity(0.15), radius: 4, y: 1)
                }
                .buttonStyle(.plain)
                .padding(12)
                .accessibilityIdentifier("map.places")
            }
        }
    }

    // MARK: actions

    private func pick(_ place: API.PlaceOut) {
        activePlace = place.place
        placesOpen = false
        flyCount += 1
        flyTo = MapFlyTarget(latitude: place.lat, longitude: place.lon, nonce: flyCount)
    }

    /// Points, places and the status for the current filters. Places and the status are extras: when they fail the
    /// map still shows.
    private func load() async {
        let f = filters
        let api: APIClient
        do {
            api = try Cloud.api()
        } catch {
            pointsError = error
            return
        }
        async let placesAnswer = try? api.placeList(
            category: f.category, kind: f.kind, person: f.person, tag: f.tag, live: f.live, from: f.from, to: f.to,
            favorite: f.favorite, misclicks: f.misclicks, forwards: f.forwards, received: f.received, dups: f.dups,
            sort: f.sort)
        async let statusAnswer = try? api.mapStatus()
        do {
            let answer = try await api.points(
                category: f.category, kind: f.kind, person: f.person, tag: f.tag, live: f.live, from: f.from, to: f.to,
                favorite: f.favorite, misclicks: f.misclicks, forwards: f.forwards, received: f.received,
                dups: f.dups, sort: f.sort)
            guard f == filters else { return }
            points = answer
            pointsError = nil
        } catch {
            if !(error is CancellationError) && !Task.isCancelled { pointsError = error }
        }
        let placesResult = await placesAnswer
        let statusResult = await statusAnswer
        guard f == filters else { return }
        if let placesResult { places = placesResult } else if places == nil { places = [] }
        if let statusResult { status = statusResult }
    }

    /// Starts the scan of older items for a location ("Scan now").
    private func scan() async {
        scanning = true
        defer { scanning = false }
        do {
            let answer = try await Cloud.api().refresh()
            Toaster.shared.ok(answer.started
                ? "Scanning older items for locations. Places appear as they are found."
                : "A scan is already running.")
            if let fresh = try? await Cloud.api().mapStatus() { status = fresh }
        } catch {
            Toaster.shared.error(error, doing: "start the scan")
        }
    }

    /// While a scan runs, the status is asked again every 4 seconds; when it ends, the points and places reload.
    private func followScan() async {
        guard status?.running == true else { return }
        while !Task.isCancelled {
            try? await Task.sleep(for: .seconds(4))
            if Task.isCancelled { return }
            guard let fresh = try? await Cloud.api().mapStatus() else { continue }
            status = fresh
            if !fresh.running {
                await load()
                return
            }
        }
    }
}
