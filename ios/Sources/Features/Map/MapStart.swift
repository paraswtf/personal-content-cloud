import Foundation

/// The map opened from another screen (`Route.mapAt`), as the website's /map links do: the library's filters
/// ("Show on map" beside the library's filters), or one item's spot (the viewer's Show on map), which the map flies
/// to and lists, as /map?at= does.
struct MapStart: Hashable {
    var filters = LibraryFilters()
    var latitude: Double?
    var longitude: Double?
}
