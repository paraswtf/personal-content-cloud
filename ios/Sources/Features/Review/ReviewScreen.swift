import SwiftUI

/// The Review tab (the website's pages/ReviewPage.tsx): the bins that need a decision. Mis-clicks (keep or
/// delete), forwards (keep or delete, section 29), duplicates (pick the best copy, delete the others) and person
/// guesses (confirm, reject or someone else), all of them or only those of one basis or one person.
struct ReviewScreen: View {
    @AppStorage("review.tab") private var tabRaw = ReviewTab.misclicks.rawValue
    @State private var stats: API.Stats?

    private var tab: ReviewTab { ReviewTab(rawValue: tabRaw) ?? .misclicks }

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Decisions the app left to you.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 16)
                ReviewTabBar(selection: Binding(get: { tab }, set: { tabRaw = $0.rawValue }), counts: counts)
            }
            Divider()
            Group {
                switch tab {
                case .misclicks: ReviewBinTab(bin: .misclicks)
                case .forwards: ReviewBinTab(bin: .forwards)
                case .duplicates: ReviewDuplicatesTab()
                case .guesses: ReviewGuessesTab()
                }
            }
            .id(tab)
            .frame(maxHeight: .infinity)
        }
        .navigationTitle("Review")
        .task { await loadStats() }
        .onCloudChange { Task { await loadStats() } }
    }

    private var counts: [ReviewTab: Int] {
        guard let stats else { return [:] }
        return [.misclicks: stats.misclicks, .forwards: stats.forwards, .duplicates: stats.duplicates,
                .guesses: stats.guesses]
    }

    private func loadStats() async {
        do {
            stats = try await Cloud.api().stats()
        } catch {
            // The counts are a hint; each bin shows its own error.
        }
    }
}

/// The four bins, in the website's order.
enum ReviewTab: String, CaseIterable, Identifiable {
    case misclicks, forwards, duplicates, guesses

    var id: String { rawValue }

    var title: String {
        switch self {
        case .misclicks: return "Mis-clicks"
        case .forwards: return "Forwards"
        case .duplicates: return "Duplicates"
        case .guesses: return "Guesses"
        }
    }

    var systemImage: String {
        switch self {
        case .misclicks: return "exclamationmark.triangle"
        case .forwards: return "arrowshape.turn.up.right"
        case .duplicates: return "square.on.square"
        case .guesses: return "person.2"
        }
    }
}

/// The bins as tabs with their counts.
struct ReviewTabBar: View {
    @Binding var selection: ReviewTab
    let counts: [ReviewTab: Int]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                ForEach(ReviewTab.allCases) { tab in
                    let selected = tab == selection
                    let count = counts[tab] ?? 0
                    Button {
                        selection = tab
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: tab.systemImage).imageScale(.small)
                            Text(tab.title)
                            if count > 0 {
                                Text(count.formatted())
                                    .font(.caption.monospacedDigit())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color(uiColor: .tertiarySystemFill), in: Capsule())
                            }
                        }
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(selected ? Color.primary : Color.secondary)
                        .padding(.horizontal, 10)
                        .frame(height: 44)
                        .overlay(alignment: .bottom) {
                            if selected {
                                Rectangle().fill(Color.accentColor).frame(height: 2)
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(count > 0 ? "\(tab.title), \(count)" : tab.title)
                    .accessibilityAddTraits(selected ? [.isSelected] : [])
                    .accessibilityIdentifier("review.tab.\(tab.rawValue)")
                }
            }
            .padding(.horizontal, 12)
        }
    }
}

/// The loading, failed and empty states of a bin, shown under its explanation.
struct ReviewListState<Item: Identifiable>: View {
    let pager: Pager<Item>
    let emptyTitle: String
    let emptyText: String
    var emptyImage = "checkmark.circle"

    var body: some View {
        if let error = pager.error, pager.items.isEmpty {
            ErrorState(error: error) { Task { await pager.retry() } }
                .frame(minHeight: 280)
        } else if pager.items.isEmpty {
            if pager.loaded && !pager.loading {
                EmptyState(title: emptyTitle, systemImage: emptyImage, text: emptyText)
                    .frame(minHeight: 280)
            } else {
                ProgressView()
                    .frame(maxWidth: .infinity, minHeight: 200)
            }
        } else if pager.loading {
            ProgressView()
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
        }
    }
}
