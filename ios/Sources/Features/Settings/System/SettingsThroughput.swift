import SwiftUI
import Charts

/// Settings > System > Processing speed (the website's ThroughputCard, docs/ARCHITECTURE.md section 32): one small
/// column chart per job type from GET /api/system/throughput, finished runs per minute (last hour), per hour (last
/// 24 hours) or per six hours (last 7 days), with the failures stacked on top in red. Tapping a chart shows that
/// time's numbers (tap again to go back to the period's); "Show as table" lists them. Asked again every 30 seconds while it is on screen.
struct SettingsThroughputSection: View {
    @State private var range: API.ThroughputRange = .hour
    @State private var data: API.ThroughputOut?
    @State private var loadError: Error?
    @State private var showTable = false

    private var period: SettingsSystemText.Period { SettingsSystemText.period(range) }

    var body: some View {
        Section {
            Picker("Period", selection: $range) {
                ForEach(SettingsSystemText.periods, id: \.label) { p in
                    Text(p.label).tag(p.range)
                }
            }
            .pickerStyle(.segmented)
            .accessibilityIdentifier("system.throughput.range")
            .task(id: range) {
                while !Task.isCancelled {
                    await load()
                    try? await Task.sleep(nanoseconds: 30_000_000_000)
                }
            }
            if let data {
                content(data)
            } else if let loadError {
                VStack(alignment: .leading, spacing: 8) {
                    Text(ErrorText.describe(loadError))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Button("Try again") { Task { await load() } }
                        .buttonStyle(.bordered)
                }
            } else {
                ProgressView("Loading processing speed")
                    .frame(maxWidth: .infinity)
            }
        } header: {
            SettingsSectionHeader(title: "Processing speed") { EmptyView() }
        } footer: {
            Text("Jobs the server finished, \(period.per). Kept for 14 days.")
        }
    }

    @ViewBuilder
    private func content(_ data: API.ThroughputOut) -> some View {
        let shown = SettingsSystemText.period(API.ThroughputRange(rawValue: data.range))
        let busy = data.types.filter { $0.total > 0 || (data.perMinute[$0.type] ?? 0) > 0 }
        let quiet = data.types.filter { t in !busy.contains { $0.type == t.type } }
        let stale = data.range != range.rawValue
        HStack(spacing: 16) {
            SettingsLegendSwatch(color: .accentColor, text: "Finished")
            SettingsLegendSwatch(color: .red, text: "Failed")
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .opacity(stale ? 0.6 : 1)
        if busy.isEmpty {
            Label("Nothing was processed \(shown.window).", systemImage: "gauge.with.dots.needle.33percent")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        } else {
            ForEach(busy, id: \.type) { t in
                SettingsThroughputChart(type: t, data: data, window: shown.window,
                                        rate: data.perMinute[t.type] ?? 0)
                    .opacity(stale ? 0.6 : 1)
            }
        }
        if !quiet.isEmpty && !busy.isEmpty {
            Text("No runs \(shown.window): \(quiet.map { $0.label.lowercased() }.joined(separator: ", ")).")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        Button(showTable ? "Hide the table" : "Show as table") {
            withAnimation { showTable.toggle() }
        }
        .accessibilityIdentifier("system.throughput.table")
        if showTable {
            SettingsThroughputTable(data: data)
        }
    }

    private func load() async {
        do {
            let answer = try await Cloud.api().throughput(range: range)
            data = answer
            loadError = nil
        } catch {
            if SettingsSystemView.cancelled(error) { return }
            if data == nil { loadError = error }
        }
    }
}

/// A coloured square and its word, for the legend.
struct SettingsLegendSwatch: View {
    let color: Color
    let text: String

    var body: some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 2)
                .fill(color)
                .frame(width: 10, height: 10)
                .accessibilityHidden(true)
            Text(text)
        }
    }
}

/// One job type: its name, how fast it goes now, its numbers for the period and its column chart.
struct SettingsThroughputChart: View {
    let type: API.ThroughputType
    let data: API.ThroughputOut
    let window: String
    let rate: Double

    @State private var selected: String?

    private struct Bar: Identifiable {
        let bucket: String
        let result: String
        let count: Int
        var id: String { bucket + result }
    }

    private var bars: [Bar] {
        var out: [Bar] = []
        for (i, bucket) in data.buckets.enumerated() {
            let total = i < type.counts.count ? type.counts[i] : 0
            let failed = i < type.failedCounts.count ? type.failedCounts[i] : 0
            let done = max(0, total - failed)
            if done > 0 { out.append(Bar(bucket: bucket, result: "Finished", count: done)) }
            if failed > 0 { out.append(Bar(bucket: bucket, result: "Failed", count: failed)) }
        }
        return out
    }

    private var top: Int { SettingsSystemText.niceMax(type.counts.max() ?? 0) }

    private var range: API.ThroughputRange { API.ThroughputRange(rawValue: data.range) }

    /// The first, middle and last buckets carry a label; the last one says "now".
    private var axisBuckets: [String] {
        let n = data.buckets.count
        guard n > 0 else { return [] }
        let picks = [0, (n - 1) / 2, n - 1]
        var seen = Set<Int>()
        return picks.filter { seen.insert($0).inserted }.map { data.buckets[$0] }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(type.label)
                    .font(.subheadline.weight(.semibold))
                Spacer(minLength: 8)
                Text(SettingsSystemText.rateLine(rate))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Text(selected.map(selectionLine) ?? SettingsSystemText.typeSummary(type, window: window))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            chart
                .frame(height: 120)
        }
        .padding(.vertical, 4)
        .accessibilityElement(children: .contain)
    }

    private var chart: some View {
        Chart {
            if let selected {
                RuleMark(x: .value("Time", selected))
                    .foregroundStyle(Color.gray.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 2))
            }
            ForEach(bars) { bar in
                BarMark(x: .value("Time", bar.bucket), y: .value("Jobs", bar.count))
                    .foregroundStyle(by: .value("Result", bar.result))
            }
        }
        .chartForegroundStyleScale(["Finished": Color.accentColor, "Failed": Color.red])
        .chartLegend(.hidden)
        .chartXScale(domain: data.buckets)
        .chartYScale(domain: 0...top)
        .chartXAxis {
            AxisMarks(values: axisBuckets) { value in
                AxisValueLabel {
                    if let bucket = value.as(String.self) {
                        Text(bucket == data.buckets.last ? "now" : SettingsSystemText.axisLabel(bucket, range: range))
                    }
                }
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: top > 1 && top % 2 == 0 ? [0, top / 2, top] : [0, top]) { _ in
                AxisGridLine()
                AxisValueLabel()
            }
        }
        // a tap, not a drag, picks a time: a drag on the chart must still scroll the screen
        .chartGesture { proxy in
            SpatialTapGesture().onEnded { value in
                let bucket: String? = proxy.value(atX: value.location.x, as: String.self)
                selected = bucket == selected ? nil : bucket
            }
        }
        .accessibilityLabel("\(type.label): \(SettingsSystemText.typeSummary(type, window: window)).")
    }

    /// The touched time's numbers, in place of the period's summary: "3 finished, 1 failed, 10:10 to 10:11".
    private func selectionLine(_ bucket: String) -> String {
        let i = data.buckets.firstIndex(of: bucket) ?? 0
        let total = i < type.counts.count ? type.counts[i] : 0
        let failed = i < type.failedCounts.count ? type.failedCounts[i] : 0
        var text = "\(max(0, total - failed).formatted()) finished"
        if failed > 0 { text += ", \(failed.formatted()) failed" }
        let when = SettingsSystemText.bucketLabel(bucket, bucketSeconds: data.bucketS, withDay: range == .week)
        return "\(text), \(when)"
    }
}

/// Every number of the charts: the times that had runs, newest first, with each job type's count.
struct SettingsThroughputTable: View {
    let data: API.ThroughputOut

    private struct Row: Identifiable {
        let index: Int
        let bucket: String
        var id: Int { index }
    }

    private var rows: [Row] {
        data.buckets.enumerated().compactMap { i, bucket in
            data.types.contains { i < $0.counts.count && $0.counts[i] > 0 } ? Row(index: i, bucket: bucket) : nil
        }
        .reversed()
    }

    var body: some View {
        let withDay = data.range == "week"
        let list = rows
        if list.isEmpty {
            Text("No runs in this period.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        } else {
            ForEach(list) { row in
                VStack(alignment: .leading, spacing: 2) {
                    Text(SettingsSystemText.bucketLabel(row.bucket, bucketSeconds: data.bucketS, withDay: withDay))
                        .font(.caption.weight(.semibold))
                    Text(line(row.index))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .accessibilityElement(children: .combine)
            }
            Text("The other times had no runs.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func line(_ i: Int) -> String {
        data.types.compactMap { t -> String? in
            let count = i < t.counts.count ? t.counts[i] : 0
            guard count > 0 else { return nil }
            let failed = i < t.failedCounts.count ? t.failedCounts[i] : 0
            return "\(t.label) \(count.formatted())" + (failed > 0 ? " (\(failed) failed)" : "")
        }
        .joined(separator: ", ")
    }
}
