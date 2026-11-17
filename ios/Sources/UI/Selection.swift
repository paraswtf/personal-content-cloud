import SwiftUI
import Observation

/// Which rows are picked on a screen that can select several (the library, an album, a person's photos).
@MainActor
@Observable
final class SelectionModel<ID: Hashable> {
    /// Selecting: taps pick rows instead of opening them.
    var active = false
    private(set) var selected: Set<ID> = []

    var count: Int { selected.count }
    var isEmpty: Bool { selected.isEmpty }

    func begin(with id: ID? = nil) {
        active = true
        if let id { selected.insert(id) }
    }

    func toggle(_ id: ID) {
        if selected.contains(id) { selected.remove(id) } else { selected.insert(id) }
    }

    func contains(_ id: ID) -> Bool { selected.contains(id) }

    func select(_ ids: [ID]) { selected.formUnion(ids) }

    func deselect(_ ids: [ID]) { selected.subtract(ids) }

    func clear() { selected.removeAll() }

    /// Leaves selecting and forgets the picks.
    func end() {
        active = false
        selected.removeAll()
    }
}

/// One action of the bulk bar.
struct BulkAction: Identifiable {
    let id: String
    let title: String
    let systemImage: String
    var role: ButtonRole?
    var disabled = false
    let run: () -> Void

    init(_ title: String, systemImage: String, role: ButtonRole? = nil, disabled: Bool = false, run: @escaping () -> Void) {
        id = title
        self.title = title
        self.systemImage = systemImage
        self.role = role
        self.disabled = disabled
        self.run = run
    }
}

/// The bar shown while selecting: how many are picked, and what can be done with them.
struct BulkBar: View {
    let count: Int
    let actions: [BulkAction]
    var onSelectAll: (() -> Void)?
    let onDone: () -> Void

    var body: some View {
        HStack(spacing: 16) {
            Button("Done", action: onDone)
                .fontWeight(.semibold)
                .accessibilityIdentifier("bulk.done")
            if let onSelectAll {
                Button("All", action: onSelectAll)
            }
            Spacer()
            Text(count == 0 ? "Select items" : Fmt.count(count, "selected", "selected"))
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("bulk.count")
            Spacer()
            ForEach(actions) { action in
                Button(role: action.role, action: action.run) {
                    Image(systemName: action.systemImage)
                        .frame(width: 32, height: 32)
                }
                .disabled(count == 0 || action.disabled)
                .accessibilityLabel(action.title)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }
}
