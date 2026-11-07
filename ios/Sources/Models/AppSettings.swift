import Foundation

/// Preferences in UserDefaults (nothing secret: the token is in the Keychain).
enum AppSettings {
    private static let defaults = UserDefaults.standard

    static var serverURL: URL? {
        get { defaults.string(forKey: "serverURL").flatMap(URL.init(string:)) }
        set { defaults.set(newValue?.absoluteString, forKey: "serverURL") }
    }

    static var wifiOnly: Bool {
        get { defaults.object(forKey: "wifiOnly") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "wifiOnly") }
    }

    static var chargingOnly: Bool {
        get { defaults.object(forKey: "chargingOnly") as? Bool ?? false }
        set { defaults.set(newValue, forKey: "chargingOnly") }
    }

    static var includeVideos: Bool {
        get { defaults.object(forKey: "includeVideos") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "includeVideos") }
    }

    static var autoSync: Bool {
        get { defaults.object(forKey: "autoSync") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "autoSync") }
    }

    /// Assets created before this date are not considered (nil = the whole library).
    static var syncSince: Date? {
        get { defaults.object(forKey: "syncSince") as? Date }
        set { defaults.set(newValue, forKey: "syncSince") }
    }

    static var lastSyncAt: Date? {
        get { defaults.object(forKey: "lastSyncAt") as? Date }
        set { defaults.set(newValue, forKey: "lastSyncAt") }
    }
}
