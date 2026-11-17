import SwiftUI
import UIKit
import PassKit

/// The virtual document of an item (the website's components/documents/VirtualDocument.tsx, section 30): its kind,
/// every field read from it, editable in place (saved as checked by the owner), hard-to-read values shaded, the
/// barcode, Reparse, Add to Apple Wallet for passes, and for a traffic challan its status, "Pay or check status" and
/// "Mark as paid".
struct ViewerDocumentCard: View {
    let context: ViewerPageContext
    let document: API.DocumentOut

    @Environment(\.openURL) private var openURL
    @State private var types: [API.DocumentTypeOut] = []
    @State private var editing: String?
    @State private var draft = ""
    @State private var draftDate = Date()
    @State private var saving: String?
    @State private var copied = false
    @State private var pass: ViewerPass?
    @State private var fetchingPass = false

    private var isChallan: Bool { document.docType == "traffic_challan" }
    private var keys: [String] {
        ViewerFormat.orderedKeys(fieldOrder: document.fieldOrder, fieldKeys: Array(document.fields.keys),
                                 docType: document.docType).filter { $0 != "pay_url" }
    }
    private var challanNumber: String { document.fields["challan_number"]?.value.trimmingCharacters(in: .whitespaces) ?? "" }
    private var paid: Bool {
        document.fields["payment_status"]?.value.trimmingCharacters(in: .whitespaces).lowercased() == "paid"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Picker("Document type", selection: Binding(
                get: { document.docType },
                set: { value in
                    guard value != document.docType else { return }
                    Task { await save("doc_type", API.DocumentUpdate(docType: value)) }
                })) {
                ForEach(typeOptions, id: \.value) { option in
                    Text(option.label).tag(option.value)
                }
            }
            .pickerStyle(.menu)
            .accessibilityIdentifier("viewer.document.type")
            HStack(alignment: .top, spacing: 12) {
                if let photo = document.photoUrl {
                    RemoteImage(ref: photo, maxPixel: 384, vault: context.vault)
                        .frame(width: 72, height: 96)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .accessibilityLabel("Photo on the document")
                }
                VStack(alignment: .leading, spacing: 6) {
                    value(key: "title", label: "Title", value: document.title ?? "", placeholder: "Not found")
                    value(key: "person_name", label: "Name", value: document.personName ?? "", placeholder: "Not found")
                    value(key: "doc_date", label: "Date", value: document.docDate ?? "",
                          shown: Fmt.plainDate(document.docDate), placeholder: "No date", isDate: true)
                }
            }
            Divider()
            if keys.isEmpty {
                Text("No fields were read from this document yet.").font(.subheadline).foregroundStyle(.secondary)
            } else {
                ForEach(keys, id: \.self) { key in
                    let field = document.fields[key]
                    let label = ViewerFormat.fieldLabel(key: key, serverLabel: field?.label, docType: document.docType)
                    value(key: "f:\(key)", label: label, value: field?.value ?? "",
                          shown: ViewerFormat.displayValue(key: key, value: field?.value ?? ""),
                          confidence: field?.confidence, mono: ViewerFormat.isMonospaced(key),
                          status: isChallan && key == "payment_status")
                }
                if keys.contains(where: { (document.fields[$0]?.confidence ?? 1) < ViewerFormat.lowConfidence }) {
                    HStack(spacing: 6) {
                        RoundedRectangle(cornerRadius: 3).fill(Color.orange.opacity(0.25)).frame(width: 12, height: 12)
                        Text("Shaded values were hard to read. Check them.").font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            if let barcode = document.barcode, !barcode.isEmpty {
                barcodeBox(barcode)
            }
            if isChallan { challanActions }
            Divider()
            HStack(spacing: 12) {
                Button {
                    Task { await reparse() }
                } label: {
                    Label("Reparse", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                walletButton
            }
            if !document.walletAvailable && document.docGroup == "pass" {
                Text("Wallet signing is not set up yet. See Settings, Wallet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
        .buttonStyle(.borderless)
        .task { types = (try? await Cloud.api().documentTypes()) ?? [] }
        .sheet(item: $pass) { pass in ViewerAddPassSheet(pass: pass.pass) }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(document.label.isEmpty ? ViewerFormat.docTypeLabel(document.docType) : document.label) details")
    }

    // MARK: parts

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: ViewerFormat.docSymbol(type: document.docType, group: document.docGroup))
                .font(.title3)
                .frame(width: 36, height: 36)
                .background(.white.opacity(0.2), in: RoundedRectangle(cornerRadius: 10))
            VStack(alignment: .leading, spacing: 2) {
                Text(ViewerFormat.groupLabel(document.docGroup).uppercased())
                    .font(.caption2.weight(.semibold))
                    .opacity(0.85)
                Text(document.title ?? (document.label.isEmpty ? ViewerFormat.docTypeLabel(document.docType) : document.label))
                    .font(.headline)
                    .lineLimit(1)
                    .accessibilityIdentifier("viewer.document.title")
                Text(document.source == .user ? "Checked by you" : "Read automatically, \(Fmt.percent(document.confidence)) sure")
                    .font(.caption)
                    .opacity(0.85)
            }
            Spacer(minLength: 0)
        }
        .foregroundStyle(.white)
        .padding(12)
        .background(LinearGradient(colors: Self.colors(document.docGroup), startPoint: .topLeading,
                                   endPoint: .bottomTrailing),
                    in: RoundedRectangle(cornerRadius: 14))
    }

    private static func colors(_ group: String) -> [Color] {
        switch group {
        case "id": return [Color(red: 0.11, green: 0.31, blue: 0.85), Color(red: 0.23, green: 0.51, blue: 0.96)]
        case "money": return [Color(red: 0.02, green: 0.47, blue: 0.34), Color(red: 0.06, green: 0.73, blue: 0.51)]
        case "education_work": return [Color(red: 0.43, green: 0.16, blue: 0.85), Color(red: 0.55, green: 0.36, blue: 0.96)]
        case "health": return [Color(red: 0.75, green: 0.07, blue: 0.24), Color(red: 0.96, green: 0.25, blue: 0.37)]
        case "pass": return [Color(red: 0.76, green: 0.25, blue: 0.05), Color(red: 0.96, green: 0.62, blue: 0.04)]
        default: return [Color(red: 0.22, green: 0.25, blue: 0.32), Color(red: 0.42, green: 0.45, blue: 0.50)]
        }
    }

    private struct TypeOption: Hashable {
        let value: String
        let label: String
    }

    private var typeOptions: [TypeOption] {
        var seen: [String: String] = [:]
        var order: [String] = []
        func add(_ value: String, _ label: String) {
            if seen[value] == nil { order.append(value); seen[value] = label }
        }
        for t in types { add(t.type, t.label.isEmpty ? ViewerFormat.docTypeLabel(t.type) : t.label) }
        for value in ["aadhaar", "pan", "passport", "driving_licence", "voter_id", "vehicle_rc", "id_card",
                      "bank_statement", "invoice", "receipt", "traffic_challan", "parking_receipt", "salary_slip",
                      "marksheet", "degree", "certificate", "offer_letter", "insurance_policy", "prescription",
                      "lab_report", "vaccination", "event_pass", "boarding_pass", "other"] {
            add(value, ViewerFormat.docTypeLabel(value))
        }
        add(document.docType, document.label.isEmpty ? ViewerFormat.docTypeLabel(document.docType) : document.label)
        return order.map { TypeOption(value: $0, label: seen[$0] ?? $0) }
    }

    /// One value with its label, a pencil to edit it in place, and the shading of a hard-to-read value.
    @ViewBuilder
    private func value(key: String, label: String, value: String, shown: String? = nil, placeholder: String = "Not found",
                       confidence: Double? = nil, mono: Bool = false, status: Bool = false, isDate: Bool = false) -> some View {
        let low = (confidence ?? 1) < ViewerFormat.lowConfidence
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text(label.uppercased()).font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                if low { Text("check this").font(.caption2).foregroundStyle(.orange) }
            }
            if editing == key {
                HStack {
                    if isDate {
                        DatePicker(label, selection: $draftDate, displayedComponents: .date)
                            .labelsHidden()
                    } else {
                        TextField(label, text: $draft)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit { Task { await commit(key: key, old: value, isDate: false) } }
                    }
                    Button {
                        Task { await commit(key: key, old: value, isDate: isDate) }
                    } label: {
                        Image(systemName: "checkmark").frame(width: 36, height: 36)
                    }
                    .accessibilityLabel("Save \(label)")
                    Button {
                        editing = nil
                    } label: {
                        Image(systemName: "xmark").frame(width: 36, height: 36)
                    }
                    .accessibilityLabel("Cancel")
                }
            } else {
                HStack(spacing: 4) {
                    if value.isEmpty {
                        Text(placeholder).foregroundStyle(.secondary)
                    } else if status {
                        Text(value)
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 3)
                            .background((ViewerFormat.isUnpaid(value) ? Color.orange : Color.secondary).opacity(0.2),
                                        in: Capsule())
                    } else {
                        Text(shown ?? value)
                            .font(mono ? .subheadline.monospaced() : .subheadline)
                            .textSelection(.enabled)
                    }
                    Spacer(minLength: 0)
                    if saving == key {
                        ProgressView().controlSize(.small)
                    } else {
                        Button {
                            draft = value
                            draftDate = Self.dayFormatter.date(from: value) ?? Date()
                            editing = key
                        } label: {
                            Image(systemName: "pencil").frame(width: 36, height: 36)
                        }
                        .accessibilityLabel("Edit \(label)")
                    }
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(low ? Color.orange.opacity(0.12) : Color.clear, in: RoundedRectangle(cornerRadius: 8))
    }

    private func barcodeBox(_ barcode: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("Barcode\(document.barcodeFormat.map { ", \($0)" } ?? "")", systemImage: "barcode")
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
            HStack(alignment: .top) {
                Text(barcode).font(.caption.monospaced()).lineLimit(6).textSelection(.enabled)
                Spacer(minLength: 0)
                Button {
                    UIPasteboard.general.string = barcode
                    copied = true
                    Task {
                        try? await Task.sleep(nanoseconds: 1_500_000_000)
                        copied = false
                    }
                } label: {
                    Image(systemName: copied ? "checkmark" : "doc.on.doc").frame(width: 36, height: 36)
                }
                .accessibilityLabel("Copy the barcode content")
            }
        }
        .padding(10)
        .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
    }

    private var challanActions: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Button {
                    if !challanNumber.isEmpty {
                        UIPasteboard.general.string = challanNumber
                        Toaster.shared.ok("Challan number copied; the portal opens next")
                    }
                    // The address was read from the document: only a web address opens, never another app's scheme.
                    let address = document.fields["pay_url"]?.value.trimmingCharacters(in: .whitespaces) ?? ""
                    let parsed: URL? = URL(string: address)
                    let isWeb: Bool = parsed?.scheme.map { ["https", "http"].contains($0.lowercased()) } ?? false
                    if let url = isWeb ? parsed : URL(string: ViewerFormat.challanPortal) { openURL(url) }
                } label: {
                    Label("Pay or check status", systemImage: "arrow.up.right.square")
                }
                .buttonStyle(.borderedProminent)
                if !paid {
                    Button {
                        Task { await markPaid() }
                    } label: {
                        Label("Mark as paid", systemImage: "checkmark")
                    }
                    .buttonStyle(.bordered)
                    .disabled(saving == "f:payment_status")
                }
            }
            Text("The portal asks for the challan number and a captcha; the number is copied when you open it. The status shown is the one printed on the challan until you mark it paid.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var walletButton: some View {
        if document.walletAvailable {
            Button {
                Task { await addToWallet() }
            } label: {
                HStack(spacing: 6) {
                    Label("Add to Apple Wallet", systemImage: "wallet.pass")
                    if fetchingPass { ProgressView().controlSize(.small) }
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(fetchingPass || !PKAddPassesViewController.canAddPasses())
            .accessibilityIdentifier("viewer.document.wallet")
        } else if document.docGroup == "pass" {
            Button {} label: {
                Label("Add to Apple Wallet", systemImage: "wallet.pass")
            }
            .buttonStyle(.bordered)
            .disabled(true)
            .accessibilityIdentifier("viewer.document.wallet")
        }
    }

    // MARK: actions

    private static let dayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = .current
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()

    private func commit(key: String, old: String, isDate: Bool) async {
        let next = isDate ? Self.dayFormatter.string(from: draftDate) : draft.trimmingCharacters(in: .whitespacesAndNewlines)
        editing = nil
        guard next != old else { return }
        let body: API.DocumentUpdate
        switch key {
        case "title": body = API.DocumentUpdate(title: next)
        case "person_name": body = API.DocumentUpdate(personName: next)
        case "doc_date": body = API.DocumentUpdate(docDate: next)
        default: body = API.DocumentUpdate(fields: [String(key.dropFirst(2)): next])
        }
        await save(key, body)
    }

    private func markPaid() async {
        await save("f:payment_status", API.DocumentUpdate(fields: ["payment_status": "Paid",
                                                                    "paid_on": Self.dayFormatter.string(from: Date())]))
    }

    private func save(_ key: String, _ body: API.DocumentUpdate) async {
        saving = key
        defer { saving = nil }
        await ViewerActions.run(context, "Saved", doing: "save the change") {
            _ = try await $0.updateDocument(uid: context.uid, body: body)
        }
    }

    private func reparse() async {
        await ViewerActions.run(context, "Reading the document again. This can take a minute.",
                                doing: "start reading it again", alwaysSay: true) {
            _ = try await $0.reparseDocument(uid: context.uid)
        }
    }

    private func addToWallet() async {
        fetchingPass = true
        defer { fetchingPass = false }
        do {
            let data = try await APIContext.$vault.withValue(context.vault) {
                try await Cloud.api().walletPass(uid: context.uid)
            }
            pass = ViewerPass(pass: try PKPass(data: data))
        } catch {
            Toaster.shared.error(error, doing: "make the Wallet pass")
        }
    }
}
