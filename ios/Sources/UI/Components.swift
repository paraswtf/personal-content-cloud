import SwiftUI

/// A person's or pet's round picture (their cover face), or their initials.
struct Avatar: View {
    let name: String?
    let ref: String?
    var size: CGFloat = 40

    var body: some View {
        ZStack {
            Circle().fill(Color(uiColor: .tertiarySystemFill))
            if let ref, !ref.isEmpty {
                RemoteImage(ref: ref, maxPixel: Int(size * 3))
                    .clipShape(Circle())
            } else {
                Text(Fmt.initials(name))
                    .font(.system(size: size * 0.38, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}

/// A small rounded label: a count, a state, a tag.
struct Chip: View {
    let text: String
    var systemImage: String?
    var tint: Color = .secondary
    var filled = false

    var body: some View {
        HStack(spacing: 4) {
            if let systemImage { Image(systemName: systemImage).imageScale(.small) }
            Text(text).lineLimit(1)
        }
        .font(.caption.weight(.medium))
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .foregroundStyle(filled ? Color.white : tint)
        .background(filled ? AnyShapeStyle(tint) : AnyShapeStyle(tint.opacity(0.14)), in: Capsule())
    }
}

/// A button-like chip that can be on or off (filters).
struct ToggleChip: View {
    let text: String
    var systemImage: String?
    @Binding var isOn: Bool

    var body: some View {
        Button {
            isOn.toggle()
        } label: {
            Chip(text: text, systemImage: systemImage, tint: .accentColor, filled: isOn)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}

/// Loading, empty and failed states for a screen's main list.
struct LoadingState: View {
    var text = "Loading"

    var body: some View {
        ProgressView(text)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct EmptyState: View {
    let title: String
    var systemImage = "tray"
    var text: String?

    var body: some View {
        ContentUnavailableView {
            Label(title, systemImage: systemImage)
        } description: {
            if let text { Text(text) }
        }
    }
}

struct ErrorState: View {
    let error: Error
    var retry: (() -> Void)?

    var body: some View {
        ContentUnavailableView {
            Label("Could not load this", systemImage: "exclamationmark.triangle")
        } description: {
            Text(ErrorText.describe(error))
        } actions: {
            if let retry {
                Button("Try again", action: retry)
                    .buttonStyle(.bordered)
            }
        }
    }
}

/// The usual body of a list screen driven by a Pager: loading, failed, empty, or the content. When a later page
/// fails, the rows stay and a bar under them says so with Try again (the pager loads no more until then).
struct PagerStateView<Item: Identifiable, Content: View>: View {
    let pager: Pager<Item>
    var emptyTitle = "Nothing here yet"
    var emptyImage = "tray"
    var emptyText: String?
    @ViewBuilder let content: () -> Content

    var body: some View {
        if pager.items.isEmpty {
            if let error = pager.error {
                ErrorState(error: error) { Task { await pager.retry() } }
            } else if pager.loaded && !pager.loading {
                EmptyState(title: emptyTitle, systemImage: emptyImage, text: emptyText)
            } else {
                LoadingState()
            }
        } else {
            content()
                .safeAreaInset(edge: .bottom) {
                    if let error = pager.error {
                        PagerMoreFailed(error: error, loading: pager.loading) { Task { await pager.retry() } }
                    }
                }
        }
    }
}

/// Under a list whose next page failed: why, and Try again.
struct PagerMoreFailed: View {
    let error: Error
    var loading = false
    let retry: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .accessibilityHidden(true)
            Text("Could not load more. \(ErrorText.describe(error))")
                .font(.footnote)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
            if loading {
                ProgressView()
            } else {
                Button("Try again", action: retry)
                    .font(.footnote.weight(.semibold))
                    .accessibilityIdentifier("pager.retry")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }
}

extension View {
    /// A confirmation for something that cannot be taken back lightly (delete, empty the trash, sign out).
    func confirmAction(_ title: String, isPresented: Binding<Bool>, actionTitle: String,
                       role: ButtonRole? = .destructive, message: String? = nil,
                       action: @escaping () -> Void) -> some View {
        confirmationDialog(title, isPresented: isPresented, titleVisibility: .visible) {
            Button(actionTitle, role: role, action: action)
            Button("Cancel", role: .cancel) {}
        } message: {
            if let message { Text(message) }
        }
    }
}
