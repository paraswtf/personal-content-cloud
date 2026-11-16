import XCTest
@testable import PersonalContentCloud

/// The vault window's arithmetic and the vault's and trash's words.
final class VaultTrashTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)

    func testSecondsLeft() {
        XCTAssertEqual(VaultClock.secondsLeft(until: nil, now: now), 0)
        XCTAssertEqual(VaultClock.secondsLeft(until: now.addingTimeInterval(-5), now: now), 0)
        XCTAssertEqual(VaultClock.secondsLeft(until: now.addingTimeInterval(845.9), now: now), 845)
    }

    func testOpenTextAndCountdown() {
        XCTAssertEqual(VaultClock.openText(secondsLeft: nil), "Unlocked")
        XCTAssertEqual(VaultClock.openText(secondsLeft: 0), "Unlocked")
        XCTAssertEqual(VaultClock.openText(secondsLeft: 845), "Unlocked for 15 min")
        XCTAssertEqual(VaultClock.openText(secondsLeft: 60), "Unlocked for 1 min")
        XCTAssertEqual(VaultClock.countdown(845), "14:05")
        XCTAssertEqual(VaultClock.countdown(9), "0:09")
        XCTAssertEqual(VaultClock.countdown(-3), "0:00")
    }

    func testUnlockErrors() {
        XCTAssertEqual(VaultText.unlockError(APIError(status: 401, detail: "invalid_credentials")),
                       "That password is not right.")
        XCTAssertTrue(VaultText.unlockError(APIError(status: 500, detail: "boom")).hasPrefix("Could not unlock."))
    }

    func testTrashWords() {
        XCTAssertEqual(TrashText.count(1), "1 item")
        XCTAssertEqual(TrashText.count(3), "3 items")
        XCTAssertEqual(TrashText.emptied(1), "1 item deleted for good.")
        XCTAssertEqual(TrashText.emptied(4), "4 items deleted for good.")
        XCTAssertEqual(TrashText.restored("IMG_1.JPG"), "IMG_1.JPG is back in the library.")
        XCTAssertEqual(TrashText.restored(nil), "Restored.")
        XCTAssertEqual(TrashText.emptyQuestion(2),
                       "All 2 items in the trash and their files will be deleted for good. This cannot be undone.")
        XCTAssertEqual(TrashText.purgeQuestion("a.jpg"),
                       "a.jpg and all its files will be deleted for good. This cannot be undone.")
        XCTAssertTrue(TrashText.intro(days: 30).hasPrefix("Deleted items stay here for 30 days"))
        XCTAssertEqual(TrashText.position(0, of: 14), "1 of 14")
    }
}

/// The seeded server's trash, config and auth state decode into the generated types, and private items stay out of
/// the trash list outside the vault (ci/seeds/vaulttrash.py).
final class VaultTrashDecodingTests: XCTestCase {
    func testTheSeededTrashAndVaultAnswersDecode() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["PCC_TEST_SERVER"], let base = URL(string: address), let token = env["PCC_TEST_TOKEN"],
              !token.isEmpty else {
            throw XCTSkip("PCC_TEST_SERVER and PCC_TEST_TOKEN are not set (the iOS workflow sets them)")
        }
        let api = APIClient(base: base, token: token)
        let config = try await api.getConfig()
        XCTAssertGreaterThan(config.trashDays, 0)
        XCTAssertGreaterThan(config.vaultUnlockMin, 0)
        let state = try await api.getState()
        XCTAssertTrue(state.authenticated)
        let page = try await api.listTrash(limit: 200)
        XCTAssertTrue(page.items.contains { $0.filename == "VAULTTRASH_KEEP.JPG" }, "the seeded trash photo is listed")
        XCTAssertFalse(page.items.contains { $0.nsfw }, "a private item is listed outside the vault")
    }
}
