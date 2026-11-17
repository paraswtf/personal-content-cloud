import XCTest
@testable import PersonalContentCloud

/// The shell's rules match the website's: the Review count badge (Nav.tsx), the vault badge's m:ss (VaultBadge.tsx),
/// the toasts' tones, durations and stack (Toast.tsx), More's headings, and what Sign out does when the server
/// does not end the session (AppShell.tsx).
@MainActor
final class ShellTests: XCTestCase {
    func testReviewBadgeHidesAtZeroAndStopsAt999() {
        XCTAssertNil(ShellStatus.badge(0))
        XCTAssertNil(ShellStatus.badge(-3))
        XCTAssertEqual(ShellStatus.badge(7), "7")
        XCTAssertEqual(ShellStatus.badge(999), "999")
        XCTAssertEqual(ShellStatus.badge(1000), "999+")
    }

    func testVaultClockReadsMinutesAndSeconds() {
        XCTAssertEqual(Fmt.clock(0), "0:00")
        XCTAssertEqual(Fmt.clock(9), "0:09")
        XCTAssertEqual(Fmt.clock(65), "1:05")
        XCTAssertEqual(Fmt.clock(14 * 60 + 59), "14:59")
        XCTAssertEqual(Fmt.clock(240 * 60), "240:00")
        XCTAssertEqual(Fmt.clock(-4), "0:00")
    }

    func testToastDurationsFollowTheWebsite() {
        XCTAssertEqual(Toaster.seconds(tone: .ok, hasAction: false), 5)
        XCTAssertEqual(Toaster.seconds(tone: .info, hasAction: false), 5)
        XCTAssertEqual(Toaster.seconds(tone: .ok, hasAction: true), 8)
        XCTAssertEqual(Toaster.seconds(tone: .error, hasAction: false), 9)
        XCTAssertEqual(Toaster.seconds(tone: .error, hasAction: true), 12)
    }

    func testToastsStackUpToFourWithTheNewestLast() {
        let toaster = Toaster.shared
        toaster.dismiss()
        defer { toaster.dismiss() }
        for n in 1...5 { toaster.ok("Message \(n)") }
        XCTAssertEqual(toaster.toasts.map(\.text), ["Message 2", "Message 3", "Message 4", "Message 5"])
        XCTAssertEqual(toaster.current?.text, "Message 5")
        toaster.info("A note")
        XCTAssertEqual(toaster.current?.tone, .info)
        XCTAssertFalse(toaster.current?.isError ?? true)
        toaster.error("It failed")
        XCTAssertTrue(toaster.current?.isError ?? false)
        toaster.dismiss(toaster.toasts.first?.id)
        XCTAssertEqual(toaster.toasts.count, 3)
        toaster.dismiss()
        XCTAssertTrue(toaster.toasts.isEmpty)
    }

    func testMoreListsEveryDestinationOnceUnderItsHeading() {
        XCTAssertEqual(MoreDestination.groups.map(\.title), ["Browse", "Tidy up", "Account"])
        let listed = MoreDestination.groups.flatMap(\.destinations)
        XCTAssertEqual(Set(listed), Set(MoreDestination.allCases))
        XCTAssertEqual(listed.count, MoreDestination.allCases.count)
    }

    func testSignOutOnlyAsksWhenTheServerCannotBeReached() {
        XCTAssertEqual(ServerSignOut.of(APIError(status: 401, detail: "not_authenticated")), .done)
        XCTAssertEqual(ServerSignOut.of(URLError(.notConnectedToInternet)), .unreachable)
        XCTAssertEqual(ServerSignOut.of(URLError(.cannotConnectToHost)), .unreachable)
        XCTAssertEqual(ServerSignOut.of(URLError(.timedOut)), .unreachable)
        XCTAssertEqual(ServerSignOut.of(APIError(status: 502, detail: "Bad Gateway")), .unreachable)
        XCTAssertEqual(ServerSignOut.of(APIError(status: 403, detail: "forbidden")), .failed)
        XCTAssertEqual(ServerSignOut.of(URLError(.cancelled)), .cancelled)
        XCTAssertEqual(ServerSignOut.of(CancellationError()), .cancelled)
    }
}
