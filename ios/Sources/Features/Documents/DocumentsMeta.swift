import SwiftUI

/// How parsed documents look (the website's components/documents/docMeta.ts): the groups' names, colours and
/// symbols, the types' names, and the few key fields a card shows, identifiers masked.
enum DocumentsMeta {
    struct Group {
        let label: String
        let systemImage: String
        let from: Color
        let to: Color
    }

    /// The groups in the website's order.
    static let groupOrder = ["id", "money", "education_work", "health", "pass", "other"]

    static func group(_ key: String?) -> Group {
        switch key {
        case "id": return Group(label: "Identity", systemImage: "person.text.rectangle", from: rgb(0x1d4ed8), to: rgb(0x3b82f6))
        case "money": return Group(label: "Money", systemImage: "banknote", from: rgb(0x047857), to: rgb(0x10b981))
        case "education_work":
            return Group(label: "Education and work", systemImage: "graduationcap", from: rgb(0x6d28d9), to: rgb(0x8b5cf6))
        case "health":
            return Group(label: "Health and insurance", systemImage: "heart.text.square", from: rgb(0xbe123c), to: rgb(0xf43f5e))
        case "pass": return Group(label: "Passes and tickets", systemImage: "ticket", from: rgb(0xc2410c), to: rgb(0xf59e0b))
        default: return Group(label: "Other", systemImage: "doc.text", from: rgb(0x374151), to: rgb(0x6b7280))
        }
    }

    static let typeLabels: [String: String] = [
        "aadhaar": "Aadhaar card", "pan": "PAN card", "passport": "Passport", "driving_licence": "Driving licence",
        "voter_id": "Voter ID", "vehicle_rc": "Vehicle RC", "id_card": "ID card", "bank_statement": "Bank statement",
        "invoice": "Invoice", "receipt": "Receipt", "traffic_challan": "Traffic challan",
        "parking_receipt": "Parking receipt", "salary_slip": "Salary slip", "marksheet": "Marksheet",
        "degree": "Degree", "certificate": "Certificate", "offer_letter": "Offer letter",
        "insurance_policy": "Insurance policy", "prescription": "Prescription", "lab_report": "Lab report",
        "vaccination": "Vaccination record", "event_pass": "Event pass", "boarding_pass": "Boarding pass",
        "other": "Document",
    ]

    /// "PAN card"; an unknown type reads as its words ("Rent agreement").
    static func typeLabel(_ type: String?) -> String {
        guard let type, !type.isEmpty else { return "Document" }
        if let known = typeLabels[type] { return known }
        let words = type.replacingOccurrences(of: "_", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }

    /// The types whose group symbol says too little get their own.
    static func typeSymbol(_ type: String?, group: String?) -> String {
        switch type {
        case "id_card": return "person.crop.rectangle"
        case "traffic_challan": return "car"
        case "parking_receipt": return "parkingsign"
        default: return Self.group(group).systemImage
        }
    }

    /// Field order and labels the app knows on its own; the server's field order and labels win when present.
    static let typeFields: [String: [(key: String, label: String)]] = [
        "traffic_challan": [
            ("challan_number", "Challan number"), ("date", "Date"), ("vehicle_number", "Vehicle number"),
            ("vehicle_owner", "Vehicle owner"), ("owner_address", "Owner address"), ("violations", "Violations"),
            ("amount", "Amount"), ("payment_status", "Payment status"), ("location", "Location"),
            ("officer", "Officer"), ("mobile", "Mobile"), ("evidences", "Evidences"), ("pay_url", "Pay at"),
        ],
        "parking_receipt": [
            ("receipt_number", "Receipt number"), ("vehicle_number", "Vehicle number"),
            ("vehicle_type", "Vehicle type"), ("in_time", "In"), ("out_time", "Out"), ("duration", "Duration"),
            ("amount", "Amount"), ("operator", "Operator"), ("location", "Location"), ("tariff", "Tariff"),
        ],
    ]

    /// The fields a card shows first for these types.
    static let keyFieldsByType: [String: [String]] = [
        "traffic_challan": ["challan_number", "amount", "payment_status"],
        "parking_receipt": ["receipt_number", "amount", "in_time"],
    ]

    private static let numberHints = ["number", "no", "pnr", "epic", "pan", "uid", "booking_id", "policy", "account", "roll"]
    private static let timeFields: Set<String> = ["in_time", "out_time"]
    private static let skipOnCard: Set<String> = ["name", "full_name", "holder_name", "address", "pay_url"]

    /// A field holding an identifier, masked on cards.
    static func isNumberField(_ key: String) -> Bool {
        let k = key.lowercased()
        return numberHints.contains { h in k == h || k.hasSuffix("_\(h)") || k.hasPrefix("\(h)_") } || k.contains("number")
    }

    /// "1234 5678 9012" to "XXXX XXXX 9012": the last four letters or digits stay, and the spacing.
    static func mask(_ value: String) -> String {
        let total = value.filter { $0.isASCII && ($0.isLetter || $0.isNumber) }.count
        guard total > 4 else { return value }
        var seen = 0
        return String(value.map { c -> Character in
            guard c.isASCII && (c.isLetter || c.isNumber) else { return c }
            seen += 1
            return seen <= total - 4 ? "X" : c
        })
    }

    /// A field's label: the server's, else the type's own, else the key.
    static func fieldLabel(_ doc: API.DocumentOut, _ key: String) -> String {
        if let label = doc.fields[key]?.label, !label.isEmpty { return label }
        if let known = typeFields[doc.docType]?.first(where: { $0.key == key }) { return known.label }
        return key
    }

    /// A value as a card shows it: the in and out times as dates with the time, the rest as stored.
    static func displayValue(_ key: String, _ value: String) -> String {
        guard timeFields.contains(key) else { return value }
        let formatted = Fmt.date(value, withTime: true)
        return formatted.isEmpty ? value : formatted
    }

    /// Field keys in display order: the server's order (else the type's own) first, then the others.
    static func orderedKeys(_ doc: API.DocumentOut) -> [String] {
        let order = doc.fieldOrder.isEmpty ? (typeFields[doc.docType] ?? []).map { $0.key } : doc.fieldOrder
        var keys = order.filter { doc.fields[$0] != nil }
        for key in doc.fields.keys.sorted() where !keys.contains(key) { keys.append(key) }
        return keys
    }

    /// Up to `limit` key fields for a card: the identifier (masked for identity documents) first, then the rest.
    static func keyFields(_ doc: API.DocumentOut, limit: Int = 3) -> [DocumentsKeyField] {
        let keys = orderedKeys(doc).filter { key in
            guard let value = doc.fields[key]?.value, !value.isEmpty else { return false }
            return !skipOnCard.contains(key)
        }
        let wanted = (keyFieldsByType[doc.docType] ?? []).filter { keys.contains($0) }
        let picked: [String]
        if !wanted.isEmpty {
            picked = wanted + keys.filter { !wanted.contains($0) }
        } else if let number = keys.first(where: isNumberField) {
            picked = [number] + keys.filter { $0 != number }
        } else {
            picked = keys
        }
        return picked.prefix(limit).map { key in
            let raw = doc.fields[key]?.value ?? ""
            let value = isNumberField(key) && doc.docGroup == "id" ? mask(raw) : displayValue(key, raw)
            return DocumentsKeyField(key: key, label: fieldLabel(doc, key), value: value)
        }
    }

    /// The card's title: the document's own, else its type's.
    static func title(_ doc: API.DocumentOut) -> String {
        if let title = doc.title, !title.isEmpty { return title }
        if !doc.label.isEmpty { return doc.label }
        return typeLabel(doc.docType)
    }

    private static func rgb(_ hex: Int) -> Color {
        Color(red: Double((hex >> 16) & 0xff) / 255, green: Double((hex >> 8) & 0xff) / 255, blue: Double(hex & 0xff) / 255)
    }
}

/// One field as a card shows it.
struct DocumentsKeyField: Identifiable, Hashable {
    let key: String
    let label: String
    let value: String

    var id: String { key }
}
