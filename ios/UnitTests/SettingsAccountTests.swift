import XCTest
import UIKit
@testable import PersonalContentCloud

/// The account side of Settings: the website's rules ported to the app (user agents, password strength and checks,
/// names, the wallet status), and the seeded server's answers these screens read.
final class SettingsAccountTests: XCTestCase {
    func testAgentsAreDescribedAsTheWebsiteDoes() {
        XCTAssertEqual(SettingsAccountText.describeAgent(nil), "Unknown device")
        XCTAssertEqual(SettingsAccountText.describeAgent(""), "Unknown device")
        XCTAssertEqual(SettingsAccountText.describeAgent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"), "Firefox on Windows")
        XCTAssertEqual(SettingsAccountText.describeAgent(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"),
                       "Safari on iPhone")
        XCTAssertEqual(SettingsAccountText.describeAgent(
            "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Mobile Safari/537.36"),
                       "Chrome on Android")
        XCTAssertEqual(SettingsAccountText.describeAgent(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36 Edg/126.0"),
                       "Edge on Mac")
        XCTAssertEqual(SettingsAccountText.describeAgent("Python-urllib/3.12"), "Script")
        XCTAssertEqual(SettingsAccountText.describeAgent("curl/8.5.0"), "Script")
        XCTAssertEqual(SettingsAccountText.describeAgent("SomethingElse/1.0"), "Browser")
        XCTAssertTrue(SettingsAccountText.isPhone("Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X)"))
        XCTAssertFalse(SettingsAccountText.isPhone("Mozilla/5.0 (Windows NT 10.0; Win64; x64)"))
        XCTAssertFalse(SettingsAccountText.isPhone(nil))
    }

    func testPasswordStrengthFollowsTheWebsite() {
        XCTAssertEqual(SettingsAccountText.passwordStrength("").score, 0)
        XCTAssertEqual(SettingsAccountText.passwordStrength("").label, "")
        XCTAssertEqual(SettingsAccountText.passwordStrength("abc").label, "Too short: use at least 10 characters.")
        XCTAssertEqual(SettingsAccountText.passwordStrength("aaaaaaaaaaaa").score, 1)  // few distinct characters
        XCTAssertEqual(SettingsAccountText.passwordStrength("abcdefghij").score, 1)    // short and one kind
        XCTAssertEqual(SettingsAccountText.passwordStrength("abcdefghijklm").score, 2)
        XCTAssertEqual(SettingsAccountText.passwordStrength("abcdefghijklmnop").score, 3)  // 16 characters
        XCTAssertEqual(SettingsAccountText.passwordStrength("Abcdefgh1234").label, "Strong.")
    }

    func testPasswordFormChecks() {
        XCTAssertEqual(SettingsAccountText.newPasswordError(current: "old", new: "short"), "Use at least 10 characters.")
        XCTAssertEqual(SettingsAccountText.newPasswordError(current: "old", new: String(repeating: "x", count: 201)),
                       "Keep it under 200 characters.")
        XCTAssertEqual(SettingsAccountText.newPasswordError(current: "same password", new: "same password"),
                       "The new password is the same as the current one.")
        XCTAssertNil(SettingsAccountText.newPasswordError(current: "old password", new: "a new password"))
        XCTAssertEqual(SettingsAccountText.matchError(new: "one", again: "two"), "The two passwords do not match.")
        XCTAssertNil(SettingsAccountText.matchError(new: "same", again: "same"))
    }

    func testNamesAreCleanedAsTheWebsiteDoes() {
        XCTAssertEqual(SettingsAccountText.cleanName("  My   new\tphone ", max: 80), "My new phone")
        XCTAssertEqual(SettingsAccountText.cleanName("abcdef", max: 3), "abc")
        XCTAssertEqual(SettingsAccountText.cleanName("   ", max: 10), "")
    }

    func testVaultLockedIsRecognised() {
        XCTAssertTrue(SettingsAccountText.isVaultLocked(APIError(status: 403, detail: "vault_locked")))
        XCTAssertFalse(SettingsAccountText.isVaultLocked(APIError(status: 403, detail: "forbidden")))
        XCTAssertFalse(SettingsAccountText.isVaultLocked(APIError(status: 401, detail: "vault_locked")))
        XCTAssertFalse(SettingsAccountText.isVaultLocked(URLError(.timedOut)))
    }

    func testWalletStatusReadsTheSystemAnswer() {
        let missing = SettingsWalletStatus(["configured": .bool(false),
                                            "missing": .array([.string("WALLET_PASS_TYPE_ID"), .string("cert.pem")])])
        XCTAssertFalse(missing.configured)
        XCTAssertEqual(missing.missing, ["WALLET_PASS_TYPE_ID", "cert.pem"])
        XCTAssertEqual(missing.dir, "DATA_DIR/wallet")
        let ready = SettingsWalletStatus(["configured": .bool(true), "missing": .array([]), "dir": .string("/srv/wallet")])
        XCTAssertTrue(ready.configured)
        XCTAssertEqual(ready.dir, "/srv/wallet")
        XCTAssertFalse(SettingsWalletStatus([:]).configured)
    }

    func testAppearanceChoices() {
        XCTAssertEqual(SettingsAppearanceChoice.allCases.map(\.label), ["System", "Light", "Dark"])
        XCTAssertEqual(SettingsAppearanceChoice.system.style, .unspecified)
        XCTAssertEqual(SettingsAppearanceChoice.light.style, .light)
        XCTAssertEqual(SettingsAppearanceChoice.dark.style, .dark)
    }

    /// The answers these screens read decode from the seeded server, and the app's own device is marked.
    func testTheSeededServersAccountAnswersDecode() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["PCC_TEST_SERVER"], let base = URL(string: address), let token = env["PCC_TEST_TOKEN"],
              !token.isEmpty else {
            throw XCTSkip("PCC_TEST_SERVER and PCC_TEST_TOKEN are not set (the iOS workflow sets them)")
        }
        let api = APIClient(base: base, token: token)
        let state = try await api.getState()
        XCTAssertEqual(state.username, "owner")
        let devices = try await api.listDevices()
        XCTAssertTrue(devices.contains { $0.current }, "the app's own device is not marked as this device")
        XCTAssertTrue(devices.allSatisfy { !SettingsAccountText.deviceLabel($0).isEmpty })
        _ = try await api.listSessions()
        _ = try await api.listPasskeys()
        let rp = try await api.relyingParty()
        XCTAssertFalse(rp.homeUrl.isEmpty)
        let system = try await api.system()
        XCTAssertNotNil(system.wallet["configured"]?.boolValue, "the wallet status has no configured flag")
    }
}
