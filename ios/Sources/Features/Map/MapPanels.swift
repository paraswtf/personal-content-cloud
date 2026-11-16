import SwiftUI

/// The panel over the bottom of the map with the items under a cluster or a point, in library order (the
/// website's "N items here" section). It is part of the screen rather than a sheet, so the viewer can open over it
/// and the panel is still there when the viewer closes.
struct MapItemsPanel: View {
    let area: MapPickedArea
    let onClose: () -> Void

    @Environment(AppRouter.self) private var router
    @State private var pager: Pager<API.ItemCard>

    init(area: MapPickedArea, filters: LibraryFilters, onClose: @escaping () -> Void) {
        self.area = area
        self.onClose = onClose
        let bbox = area.bboxParam
        _pager = State(initialValue: Pager<API.ItemCard> { cursor in
            try await Cloud.api().items(bbox: bbox, cursor: cursor, limit: 60, category: filters.category,
                                        kind: filters.kind, person: filters.person, tag: filters.tag,
                                        live: filters.live, from: filters.from, to: filters.to,
                                        favorite: filters.favorite, misclicks: filters.misclicks,
                                        forwards: filters.forwards, received: filters.received,
                                        dups: filters.dups, sort: filters.sort).pagerPage
        })
    }

    private var title: String {
        let total = pager.total ?? 0
        return "\(Fmt.count(total > 0 ? total : area.count, "item")) here"
    }

    var body: some View {
        VStack(spacing: 0) {
            Capsule()
                .fill(Color.secondary.opacity(0.4))
                .frame(width: 36, height: 5)
                .padding(.top, 6)
                .accessibilityHidden(true)
            HStack(spacing: 8) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .accessibilityIdentifier("map.panel.title")
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .font(.system(size: 14, weight: .semibold))
                        .frame(width: 36, height: 36)
                        .background(Color(uiColor: .tertiarySystemFill), in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Close")
                .accessibilityIdentifier("map.panel.close")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
            PagerStateView(pager: pager, emptyTitle: "Nothing here", emptyImage: "mappin.slash") {
                MediaGrid(items: pager.items, sections: .none,
                          onOpen: { item in open(item) },
                          onAppear: { item in pager.loadMoreIfNeeded(item) })
            }
        }
        .background(Color(uiColor: .systemBackground))
        .clipShape(UnevenRoundedRectangle(topLeadingRadius: 18, topTrailingRadius: 18))
        .shadow(color: .black.opacity(0.18), radius: 10, y: -2)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Items here")
        .task { if !pager.loaded { await pager.refresh() } }
        .onCloudChange { Task { await pager.refresh() } }
    }

    private func open(_ item: API.ItemCard) {
        router.show(pager.items, start: item) {
            await pager.loadMore()
            return pager.items.map(\.viewerEntry)
        }
    }
}

/// The places with photos, most photos first (the website's PlacesList, in a sheet on the phone). Picking one flies
/// the map there.
struct MapPlacesSheet: View {
    let places: [API.PlaceOut]?
    let active: String?
    let onPick: (API.PlaceOut) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if let places {
                    if places.isEmpty {
                        ContentUnavailableView {
                            Label("No places yet", systemImage: "mappin.slash")
                        } description: {
                            Text("Photos with a location get a place name within a few minutes.")
                        }
                    } else {
                        list(places)
                    }
                } else {
                    ProgressView("Loading places")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .navigationTitle("Places")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private func list(_ places: [API.PlaceOut]) -> some View {
        List {
            Section {
                ForEach(places, id: \.place) { place in
                    Button {
                        onPick(place)
                    } label: {
                        row(place)
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("map.place")
                    .accessibilityAddTraits(active == place.place ? .isSelected : [])
                }
            } footer: {
                Text("Map: Apple Maps, over the internet; your photos stay on your server. Place names: GeoNames (CC BY 4.0).")
            }
        }
        .listStyle(.insetGrouped)
    }

    private func row(_ place: API.PlaceOut) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: "mappin.and.ellipse")
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(place.placeCity ?? place.place)
                    .fontWeight(active == place.place ? .semibold : .regular)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                let region = [place.placeAdmin, place.placeCountry].compactMap { $0 }.filter { !$0.isEmpty }
                if !region.isEmpty {
                    Text(region.joined(separator: ", "))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 8)
            Text(Fmt.count(place.count, "item"))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .contentShape(Rectangle())
    }
}

/// The line under the header about places that are still being found (the website's status line).
struct MapStatusBanner: View {
    let status: API.MapStatus
    let scanning: Bool
    let onScan: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !status.gazetteer {
                Text("Place names are off: the GeoNames files are missing from the models folder (see the README).")
            }
            if status.running {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Scanning older items for locations"
                         + (status.unchecked > 0 ? ", \(status.unchecked.formatted()) to go." : "."))
                }
            } else if status.unchecked > 0 {
                HStack(spacing: 8) {
                    Text("\(Fmt.count(status.unchecked, "older item")) \(status.unchecked == 1 ? "has" : "have") "
                         + "not been checked for a location yet.")
                    Spacer(minLength: 4)
                    Button {
                        onScan()
                    } label: {
                        if scanning { ProgressView().controlSize(.small) } else { Text("Scan now") }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(scanning)
                    .accessibilityIdentifier("map.scan")
                }
            }
        }
        .font(.footnote)
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .contain)
    }
}
