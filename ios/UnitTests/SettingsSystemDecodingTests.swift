import XCTest
@testable import PersonalContentCloud

/// The seeded server's system, learning and throughput answers read into what the system side of Settings shows
/// (ios/ci/seed_server.py: models off, backups on). Skipped without the workflow's server. Nothing here starts work
/// on the server: only GET requests.
final class SettingsSystemDecodingTests: XCTestCase {
    func testTheSystemScreensReadTheSeededServer() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["PCC_TEST_SERVER"], let base = URL(string: address), let token = env["PCC_TEST_TOKEN"],
              !token.isEmpty else {
            throw XCTSkip("PCC_TEST_SERVER and PCC_TEST_TOKEN are not set (the iOS workflow sets them)")
        }
        let api = APIClient(base: base, token: token)

        let system = try await api.system()
        let status = SettingsSystemStatus(system)
        XCTAssertFalse(system.models.isEmpty, "the server lists its models")
        XCTAssertGreaterThan(SettingsSystemStatus.missingModels(system), 0, "the CI server runs with the models off")
        XCTAssertFalse(status.queues.byQueue.isEmpty, "the queues are listed by name")
        XCTAssertNotNil(status.video, "the server reports its video copies")
        XCTAssertFalse(SettingsTier.tiers(system).isEmpty)
        XCTAssertEqual(system.backup?.enabled, true, "the workflow sets BACKUP_DIR")
        XCTAssertFalse(SettingsSystemText.backupLastLine(status.backupLast).isEmpty)

        let learning = try await api.learningStatus()
        let learners = SettingsLearners(learning.learners)
        XCTAssertFalse(SettingsLearningText.docTypes(learners).isEmpty)
        XCTAssertFalse(SettingsSystemText.trainedLine(learning).isEmpty)

        for range in [API.ThroughputRange.hour, .day, .week] {
            let t = try await api.throughput(range: range)
            XCTAssertEqual(t.range, range.rawValue)
            for type in t.types {
                XCTAssertEqual(type.counts.count, t.buckets.count, "\(type.type): one count per bucket")
                XCTAssertEqual(type.failedCounts.count, t.buckets.count, "\(type.type): one failure count per bucket")
            }
        }
        let hour = try await api.throughput(range: .hour)
        XCTAssertTrue(hour.types.contains { $0.type == "ingest" && $0.total > 0 }, "the seed's uploads were stored in the last hour")
    }
}
