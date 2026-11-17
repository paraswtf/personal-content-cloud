import Foundation
import CoreGraphics

/// The boxes area's writes. Each is one owner action with the website's message, and Undo when the server changed
/// something; the item and the screens showing it load again. They return whether the change went through; a
/// failure shows its error (or returns it in words, for the stage's own bars).
@MainActor
enum BoxesActions {
    // MARK: running an action

    /// Runs one action: the undo history keeps `label`, the toast says `message(answer)` (nothing for nil).
    static func run<T>(_ label: String, doing: String, uid: String, vault: Bool, message: (T) -> String?,
                       _ body: () async throws -> T) async -> T? {
        let before = UndoCenter.shared.entries.first?.id
        do {
            let value = try await APIContext.$vault.withValue(vault) {
                try await UndoCenter.shared.perform(label, toast: false) { try await body() }
            }
            if let text = message(value) { toast(text, since: before) }
            CloudEvents.post(items: [uid])
            return value
        } catch {
            Toaster.shared.error(error, doing: doing)
            // Part of a change of several boxes may have gone through: the item is read again either way.
            CloudEvents.post(items: [uid])
            return nil
        }
    }

    /// The message, with Undo when an action joined the history since `before`.
    static func toast(_ text: String, since before: String?) {
        if let entry = UndoCenter.shared.entries.first, entry.id != before {
            let id = entry.id
            Toaster.shared.ok(text, action: .init(label: "Undo") { Task { await UndoCenter.shared.undo(id) } })
        } else {
            Toaster.shared.ok(text)
        }
    }

    /// Who a pick names: an existing person's id, or a new name.
    static func parts(_ pick: PersonPick?) -> (personId: Int?, newName: String?) {
        switch pick {
        case .existing(let option)?: return (option.id, nil)
        case .new(let name, _)?: return (nil, name)
        case nil: return (nil, nil)
        }
    }

    private static func ids(_ targets: [BoxesTarget], _ kind: StageBox.Kind) -> [Int] {
        targets.filter { $0.kind == kind }.map(\.boxId)
    }

    // MARK: faces and person boxes (the website's FaceEditor)

    /// Names the faces and person boxes, or takes their person off with a nil pick.
    static func assign(_ targets: [BoxesTarget], pick: PersonPick?, current: API.PersonRef?, uid: String,
                       vault: Bool) async -> Bool {
        let faces = ids(targets, .face), bodies = ids(targets, .body)
        let (personId, newName) = parts(pick)
        let tail = BoxesWords.trackTail(targets.count)
        let unassign = pick == nil
        let result = await run(unassign ? "Took a name off a box" : "Named a box", doing: "save the name", uid: uid,
                               vault: vault, message: { (res: API.AssignOut) -> String? in
            let followed = res.followed > 0 ? ", with \(Fmt.count(res.followed, "look-alike face"))" : ""
            if unassign {
                if let current { return "Removed from \(Fmt.personName(current.name))\(followed)\(tail)" }
                return "Unassigned\(tail)"
            }
            let base = BoxesWords.name(res.person).map { "Tagged as \($0)\(followed)" } ?? "Saved"
            return BoxesWords.withGuessed(base + tail, res.guessed)
        }) {
            var res: API.AssignOut?
            if !faces.isEmpty {
                let body = API.AssignFacesIn(faceIds: faces, personId: personId, newPersonName: newName,
                                             nullKeys: unassign ? [.personId] : [])
                res = try await Cloud.api().assignFaces(body: body)
            }
            if !bodies.isEmpty {
                let body = API.AssignBodiesIn(bodyIds: bodies, personId: personId, newPersonName: newName,
                                              nullKeys: unassign ? [.personId] : [])
                let r2 = try await Cloud.api().assignBodies(body: body)
                if var r = res {
                    r.updated += r2.updated
                    r.followed += r2.followed
                    r.guessed += r2.guessed
                    if r.person == nil { r.person = r2.person }
                    res = r
                } else {
                    res = r2
                }
            }
            return res ?? API.AssignOut(updated: 0, person: nil, followed: 0, guessed: 0)
        }
        return result != nil
    }

    /// "Someone else" on a guess: the guess is learned as wrong and the box goes to the person picked, in one step.
    static func correct(_ targets: [BoxesTarget], pick: PersonPick, uid: String, vault: Bool) async -> Bool {
        let faces = ids(targets, .face), bodies = ids(targets, .body)
        let (personId, newName) = parts(pick)
        let tail = BoxesWords.trackTail(targets.count)
        let result = await run("Named a box", doing: "save the name", uid: uid, vault: vault,
                               message: { (res: API.AssignOut) -> String? in
            BoxesWords.withGuessed((BoxesWords.name(res.person).map { "Tagged as \($0)" } ?? "Saved") + tail, res.guessed)
        }) {
            try await Cloud.api().correctGuesses(body: API.GuessCorrectIn(faces: faces, bodies: bodies, personId: personId,
                                                                          newPersonName: newName))
        }
        return result != nil
    }

    /// Answers the guess on the boxes: yes, it is them, or no.
    static func answer(_ targets: [BoxesTarget], confirm: Bool, guessName: String, uid: String, vault: Bool) async -> Bool {
        let body = API.GuessIdsIn(faces: ids(targets, .face), bodies: ids(targets, .body))
        let text = (confirm ? "Tagged as \(guessName)" : "Not \(guessName)") + BoxesWords.trackTail(targets.count)
        let result = await run(confirm ? "Confirmed a guess" : "Said no to a guess", doing: "answer the guess", uid: uid,
                               vault: vault, message: { (_: API.BulkOut) -> String? in text }) {
            if confirm {
                return try await Cloud.api().confirmGuesses(body: body)
            } else {
                return try await Cloud.api().rejectGuesses(body: body)
            }
        }
        return result != nil
    }

    /// "Not a face", "Not a person", "Remove box": the boxes go and are not shown as that again.
    static func remove(_ targets: [BoxesTarget], noun: String, uid: String, vault: Bool) async -> Bool {
        let tail = BoxesWords.trackTail(targets.count)
        let result = await run("Removed a box", doing: "remove the box", uid: uid, vault: vault,
                               message: { (followed: Int) -> String? in
            let more = followed > 0 ? " \(Fmt.count(followed, "look-alike")) went with it." : ""
            return "Removed. It will not be shown as a \(noun) again.\(more)\(tail)"
        }) {
            var followed = 0
            for t in targets {
                switch t.kind {
                case .face:
                    let r = try await Cloud.api().deleteFace(faceId: t.boxId)
                    followed += r.followed
                case .body:
                    _ = try await Cloud.api().deleteBody(bodyId: t.boxId)
                case .animal:
                    _ = try await Cloud.api().deleteAnimal(animalId: t.boxId)
                }
            }
            return followed
        }
        return result != nil
    }

    /// A tattoo, nails or mehndi seen on the person: not one at all, not theirs, or shown again (`verdict` nil).
    static func trait(_ trait: API.TraitOut, verdict: API.Verdict?, owner: String?, uid: String, vault: Bool) async -> Bool {
        let word = BoxesWords.traitWord(trait.kind)
        let text: String
        if verdict == nil {
            text = "Shown again on this photo: \(trait.label)."
        } else if verdict == .notTrait {
            text = "Not counted as \(trait.kind == .tattoo ? "a tattoo" : word) any more: \(trait.label)."
        } else {
            text = "Not counted as \(owner.map { "\($0)'s" } ?? "this person's") \(word) any more: \(trait.label)."
        }
        let result = await run("Changed what is seen on a person", doing: "save that", uid: uid, vault: vault,
                               message: { (_: API.TraitOut) -> String? in text }) {
            try await Cloud.api().hideTrait(traitId: trait.id, body: API.TraitHideIn(hidden: verdict != nil, verdict: verdict))
        }
        return result != nil
    }

    // MARK: pets (the website's AnimalEditor)

    /// Names the pet boxes (or takes the pet off with a nil pick). `fixSpecies` makes the picked pet this box's
    /// species first.
    static func assignPet(_ targets: [BoxesTarget], pick: PersonPick?, fixSpecies: Bool, current: API.PersonRef?,
                          species: String?, uid: String, vault: Bool) async -> Bool {
        let animals = ids(targets, .animal)
        let (personId, newName) = parts(pick)
        let tail = BoxesWords.trackTail(targets.count)
        let unassign = pick == nil
        let result = await run(unassign ? "Took a name off a pet box" : "Named a pet", doing: "save the name", uid: uid,
                               vault: vault, message: { (res: API.AssignOut) -> String? in
            if unassign {
                if let current { return "Removed from \(Fmt.personName(current.name, kind: "pet"))\(tail)" }
                return "Unassigned\(tail)"
            }
            if fixSpecies, let name = BoxesWords.name(res.person) {
                return "Tagged as \(name), now \(BoxesWords.aSpecies(species))\(tail)"
            }
            return (BoxesWords.name(res.person).map { "Tagged as \($0)" } ?? "Saved") + tail
        }) {
            try await Cloud.api().assignAnimals(body: API.AssignAnimalsIn(animalIds: animals, personId: personId,
                                                                          newPersonName: newName,
                                                                          fixSpecies: fixSpecies ? true : nil,
                                                                          nullKeys: unassign ? [.personId] : []))
        }
        return result != nil
    }

    /// "Not a dog": what species the pet boxes really are; look-alike crops follow.
    static func setSpecies(_ targets: [BoxesTarget], species: API.Species, uid: String, vault: Bool) async -> Bool {
        let animals = ids(targets, .animal)
        let label = Fmt.speciesLabel(species.rawValue).lowercased()
        let result = await run("Changed a pet's species", doing: "change the species", uid: uid, vault: vault,
                               message: { (res: API.AssignOut) -> String? in
            let crops = res.followed > 0 ? " \(Fmt.count(res.followed, "look-alike crop")) followed." : ""
            return "Changed to \(label).\(crops)"
        }) {
            try await Cloud.api().setAnimalSpecies(body: API.SpeciesIn(animalIds: animals, species: species))
        }
        return result != nil
    }

    // MARK: moving and adding boxes

    /// Saves a moved or resized box. Returns nil when it was saved, else what went wrong in words.
    static func move(_ box: StageBox, to rect: CGRect, uid: String, vault: Bool) async -> String? {
        let label: String
        switch box.kind {
        case .face: label = "Face box moved"
        case .body: label = "Person box moved"
        case .animal: label = "Pet box moved"
        }
        let body = API.BoxIn(x: Double(rect.minX), y: Double(rect.minY), w: Double(rect.width), h: Double(rect.height))
        do {
            try await APIContext.$vault.withValue(vault) {
                try await UndoCenter.shared.perform(label) {
                    switch box.kind {
                    case .face: _ = try await Cloud.api().moveFace(faceId: box.boxId, body: body)
                    case .body: _ = try await Cloud.api().moveBody(bodyId: box.boxId, body: body)
                    case .animal: _ = try await Cloud.api().moveAnimal(animalId: box.boxId, body: body)
                    }
                }
            }
            CloudEvents.post(items: [uid])
            return nil
        } catch {
            if (error as? APIError)?.detail == "no_face" {
                return "No face there. For someone whose face does not show, use Add a person box instead."
            }
            return ErrorText.describe(error, doing: "move the box")
        }
    }

    /// Adds the person box or face box the owner drew, with who is in it. Returns nil when it was added, else what
    /// went wrong in words.
    static func addBox(_ kind: StageDrawKind, rect: CGRect, pick: PersonPick, uid: String, vault: Bool) async -> String? {
        let (personId, newName) = parts(pick)
        let typed = newName ?? "them"
        let x = Double(rect.minX), y = Double(rect.minY), w = Double(rect.width), h = Double(rect.height)
        let before = UndoCenter.shared.entries.first?.id
        do {
            let text: String = try await APIContext.$vault.withValue(vault) {
                try await UndoCenter.shared.perform(kind == .face ? "Added a face box" : "Added a person box", toast: false) {
                    if kind == .face {
                        let r = try await Cloud.api().addFaceBox(uid: uid, body: API.FaceBoxIn(x: x, y: y, w: w, h: h,
                                                                                                personId: personId,
                                                                                                newPersonName: newName))
                        let who = Fmt.isNamed(r.person?.name) ? Fmt.personName(r.person?.name) : typed
                        return "Tagged \(who) with a face box."
                    }
                    let r = try await Cloud.api().addPersonBox(uid: uid, body: API.PersonBoxIn(x: x, y: y, w: w, h: h,
                                                                                                personId: personId,
                                                                                                newPersonName: newName))
                    let who = Fmt.isNamed(r.body.person?.name) ? Fmt.personName(r.body.person?.name) : typed
                    if r.face != nil { return "Tagged \(who). Their face in the box is tagged too." }
                    return "Tagged \(who) with a person box."
                }
            }
            toast(text, since: before)
            CloudEvents.post(items: [uid])
            return nil
        } catch {
            return BoxesWords.addFailure(error)
        }
    }

    /// Adds the pet box the owner drew, with its species. Returns the new box, or nil after showing the error.
    static func addPet(rect: CGRect, species: API.Species, uid: String, vault: Bool) async -> API.AnimalOut? {
        let body = API.AnimalBoxIn(x: Double(rect.minX), y: Double(rect.minY), w: Double(rect.width),
                                   h: Double(rect.height), species: species.rawValue)
        let what = BoxesWords.aSpecies(species.rawValue)
        return await run("Added a pet box", doing: "add the box", uid: uid, vault: vault,
                         message: { (_: API.AnimalOut) -> String? in "Added \(what) box" }) {
            try await Cloud.api().addAnimalBox(uid: uid, body: body)
        }
    }

    /// "Find pets again": the detector looks at the item again, more closely.
    static func findPets(uid: String, vault: Bool) async {
        let before = UndoCenter.shared.entries.first?.id
        do {
            let found = try await APIContext.$vault.withValue(vault) {
                try await UndoCenter.shared.perform("Looked for pets again", toast: false) {
                    try await Cloud.api().redetectAnimals(uid: uid)
                }
            }
            toast(found.isEmpty ? "No pet found" : "Added \(Fmt.count(found.count, "pet box", "pet boxes"))", since: before)
            CloudEvents.post(items: [uid])
        } catch {
            if let api = error as? APIError, api.status == 409 {
                Toaster.shared.error("The models are not loaded")
            } else {
                Toaster.shared.error(error, doing: "look for pets")
            }
        }
    }

    // MARK: people on the item (the website's PersonEdit and the info panel's tags)

    /// "Change to...": their faces, boxes and tags here go to someone else.
    static func change(_ who: API.PersonRef, to pick: PersonPick, uid: String, vault: Bool) async -> Bool {
        let (personId, newName) = parts(pick)
        let pet = who.kind == .pet
        let name = BoxesWords.displayName(who.name, pet: pet)
        let result = await run("Changed who is in an item", doing: "change them", uid: uid, vault: vault,
                               message: { (r: API.PeopleBulkOut) -> String? in
            guard r.updated > 0 else { return "\(name) is not on this item: nothing changed." }
            let to = BoxesWords.name(r.person) ?? "someone else"
            return "Changed to \(to)." + BoxesWords.followedTail(followed: r.followed, pulled: r.pulled, pet: pet)
        }) {
            try await Cloud.api().movePeopleOnItems(body: API.PeopleMoveIn(uids: [uid], fromPersonId: who.id,
                                                                           personId: personId, newPersonName: newName))
        }
        return result != nil
    }

    /// "Rename": the name changes everywhere; a name someone else has merges them.
    static func rename(_ who: API.PersonRef, to name: String?, uid: String, vault: Bool) async -> Bool {
        let hadName = Fmt.isNamed(who.name)
        let result = await run(name == nil ? "Took a name off" : "Renamed someone", doing: "save the name", uid: uid,
                               vault: vault, message: { (p: API.PersonOut) -> String? in
            if p.id != who.id { return "Merged into \(BoxesWords.name(p) ?? "them")." }
            if let n = p.name, Fmt.isNamed(n) { return hadName ? "Renamed to \(n)." : "Named \(n)." }
            return "Name removed."
        }) {
            try await Cloud.api().updatePerson(personId: who.id, body: API.PersonUpdate(name: name,
                                                                                         nullKeys: name == nil ? [.name] : []))
        }
        return result != nil
    }

    /// "Not NAME": their boxes here stay, unnamed and remembered as not them, and their tags here go.
    static func notThem(_ who: API.PersonRef, uid: String, vault: Bool) async -> Bool {
        let pet = who.kind == .pet
        let name = BoxesWords.displayName(who.name, pet: pet)
        let result = await run("Took someone off an item", doing: "remove them", uid: uid, vault: vault,
                               message: { (r: API.PeopleBulkOut) -> String? in
            guard r.updated > 0 else { return "\(name) is not on this item: nothing changed." }
            return "Not \(name): removed from this item." + BoxesWords.followedTail(followed: r.followed, pulled: r.pulled, pet: pet)
        }) {
            try await Cloud.api().removePeopleFromItems(body: API.PeopleRemoveIn(uids: [uid], personId: who.id))
        }
        return result != nil
    }

    /// "Not a face", "Not a person", "Not a pet" for everything of theirs of that kind here, as one action.
    /// `stays`: a face taken away sat in a person box of theirs, which keeps them on the item.
    static func removeBoxes(of who: API.PersonRef, kind: StageBox.Kind, ids: [Int], stays: Bool, uid: String,
                            vault: Bool) async -> Bool {
        let name = BoxesWords.displayName(who.name, pet: who.kind == .pet)
        let what: String, plural: String
        switch kind {
        case .face: (what, plural) = ("face", "faces")
        case .body: (what, plural) = ("person box", "person boxes")
        case .animal: (what, plural) = ("pet box", "pet boxes")
        }
        let boxes = ids.count == 1 ? "\(name)'s \(what)" : "\(name)'s \(ids.count) \(plural)"
        let result = await run("Removed boxes", doing: "remove the boxes", uid: uid, vault: vault,
                               message: { (followed: Int) -> String? in
            switch kind {
            case .face:
                let tail = followed > 0 ? " \(Fmt.count(followed, "look-alike face")) went with \(ids.count == 1 ? "it" : "them")." : ""
                let stay = stays ? " \(name) stays here through their person box." : ""
                return "Removed \(boxes) as not a face.\(tail)\(stay)"
            case .body:
                return "Removed \(boxes): not a person."
            case .animal:
                return "Removed \(boxes): not a pet."
            }
        }) {
            var followed = 0
            for id in ids {
                switch kind {
                case .face:
                    let r = try await Cloud.api().deleteFace(faceId: id)
                    followed += r.followed
                case .body:
                    _ = try await Cloud.api().deleteBody(bodyId: id)
                case .animal:
                    _ = try await Cloud.api().deleteAnimal(animalId: id)
                }
            }
            return followed
        }
        return result != nil
    }

    /// "Species...": what kind of animal a pet is, everywhere (its boxes follow).
    static func personSpecies(_ who: API.PersonRef, species: API.Species, uid: String, vault: Bool) async -> Bool {
        let label = Fmt.speciesLabel(species.rawValue).lowercased()
        let result = await run("Changed a pet's species", doing: "change the species", uid: uid, vault: vault,
                               message: { (r: API.AssignOut) -> String? in
            let petName = BoxesWords.displayName(r.person?.name ?? who.name, pet: true)
            var joined = ""
            if let p = r.person, p.id != who.id { joined = " It joined \(BoxesWords.displayName(p.name, pet: true))." }
            let crops = r.followed > 0 ? " \(Fmt.count(r.followed, "look-alike crop")) followed." : ""
            return "\(petName) is a \(label) now.\(joined)\(crops)"
        }) {
            try await Cloud.api().setPersonSpecies(personId: who.id, body: API.PersonSpeciesIn(species: species))
        }
        return result != nil
    }

    /// "Tag someone": a person or pet in the item, or (`related`) one the item is about without them in it. A new
    /// pet comes with its species.
    static func tag(_ pick: PersonPick, related: Bool, species: String?, uid: String, vault: Bool) async -> Bool {
        let (personId, newName) = parts(pick)
        var kind: API.ItemPersonOutKind?
        if case .new(_, let k) = pick, k == "pet" { kind = .pet }
        let body = API.TagIn(personId: personId, newPersonName: newName, kind: kind, species: kind == .pet ? species : nil,
                             relation: API.Relation(rawValue: related ? "related" : "in"))
        let result = await run(related ? "Tagged someone as related" : "Tagged someone", doing: "add the tag", uid: uid,
                               vault: vault, message: { (_: [API.PersonRef]) -> String? in
            related ? "Tagged as related" : "Tag added"
        }) {
            try await Cloud.api().addTag(uid: uid, body: body)
        }
        return result != nil
    }

    /// "Make this 'in'" / "Make this 'related'".
    static func setRelation(_ who: API.PersonRef, related: Bool, uid: String, vault: Bool) async -> Bool {
        let body = API.TagIn(personId: who.id, relation: API.Relation(rawValue: related ? "related" : "in"))
        let result = await run("Changed a tag", doing: "change the tag", uid: uid, vault: vault,
                               message: { (_: [API.PersonRef]) -> String? in
            related ? "Marked as related: not in the picture" : "Marked as in the picture"
        }) {
            try await Cloud.api().addTag(uid: uid, body: body)
        }
        return result != nil
    }

    /// "Remove from this item" for someone here through a tag alone.
    static func removeTag(_ who: API.PersonRef, uid: String, vault: Bool) async -> Bool {
        let result = await run("Removed a tag", doing: "remove the tag", uid: uid, vault: vault,
                               message: { (_: [API.PersonRef]) -> String? in "Tag removed" }) {
            try await Cloud.api().removeTag(uid: uid, personId: who.id)
        }
        return result != nil
    }
}
