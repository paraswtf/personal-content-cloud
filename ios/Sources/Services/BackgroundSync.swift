import Foundation
import BackgroundTasks
import Photos

/// Background runs: a BGProcessingTask when iOS allows it (network needed, power optional), and a
/// library change observer that queues a run while the app is open.
enum BackgroundSync {
    static let identifier = "in.rivant.pcc.sync"
    private static var observer: ChangeObserver?

    static func register() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: identifier, using: nil) { task in
            guard let task = task as? BGProcessingTask else { return }
            handle(task)
        }
        observer = ChangeObserver()
        PHPhotoLibrary.shared().register(observer!)
        schedule()
    }

    static func schedule() {
        guard AppSettings.autoSync else { return }
        let req = BGProcessingTaskRequest(identifier: identifier)
        req.requiresNetworkConnectivity = true
        req.requiresExternalPower = AppSettings.chargingOnly
        req.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)
        try? BGTaskScheduler.shared.submit(req)
    }

    private static func handle(_ task: BGProcessingTask) {
        schedule()
        let deadline = Date(timeIntervalSinceNow: 25)   // keep well inside what iOS grants
        Task { @MainActor in
            let engine = SyncEngine.shared
            task.expirationHandler = { engine.cancel() }
            engine.start(manual: false, deadline: deadline)
            while engine.running { try? await Task.sleep(nanoseconds: 500_000_000) }
            task.setTaskCompleted(success: engine.lastError == nil)
        }
    }

    final class ChangeObserver: NSObject, PHPhotoLibraryChangeObserver {
        func photoLibraryDidChange(_ changeInstance: PHChange) {
            guard AppSettings.autoSync else { return }
            Task { @MainActor in SyncEngine.shared.start(manual: false) }
        }
    }
}
