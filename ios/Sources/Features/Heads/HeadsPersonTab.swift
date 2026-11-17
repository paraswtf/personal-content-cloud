import SwiftUI

/// A person's 3D tab (the website's components/people/Model3dTab.tsx; docs/ARCHITECTURE.md section 36): a 3D head
/// for each look found in their photos. Before a model exists it says what it does and shows the options and
/// "Generate"; nothing is generated before that. While the server works it shows the step and "Stop". With a model
/// it shows the head and the sliders that pick the look, turn the head and move the light, and under the head what
/// the owner is looking at. The status is read again each time the tab is shown, every 2 seconds while a model is
/// made, every 5 while someone else's is, and when a head does not load (its look may be gone on the server).
///
/// The people area shows it as the person page's 3D tab. It has no scroll view of its own: the person page's
/// scroll view holds it, and `reveal` scrolls it to a view of this tab by its id (the Generate again panel, opened
/// from the line above the head, is brought into view). The head itself takes a drag that starts on it, so the
/// page scrolls from around it.
struct HeadsPersonTab: View {
    let personId: Int
    let name: String
    var reveal: ((String) -> Void)? = nil

    /// The id of the Generate again panel, for `reveal`.
    static let againPanelId = "heads.againPanel"

    @State private var status: API.Model3dOut?
    @State private var loadError: Error?
    @State private var options = HeadsOptions.defaults
    @State private var again = false
    @State private var confirmDelete = false
    @State private var starting = false
    @State private var stopping = false
    @State private var refresh = 0
    /// A head or a look's picture did not load and the status is being read again: the stage says "Checking the
    /// model" until it is back.
    @State private var checking = false
    @State private var askedAt = Date.distantPast
    @State private var store: HeadsStore

    init(personId: Int, name: String, reveal: ((String) -> Void)? = nil) {
        self.personId = personId
        self.name = name
        self.reveal = reveal
        _store = State(initialValue: HeadsStore(personId: personId))
    }

    private var whose: String { name.isEmpty ? "this person's" : "\(name)'s" }

    var body: some View {
        Group {
            if let status {
                content(status)
            } else if let loadError {
                VStack(spacing: 12) {
                    Text(ErrorText.describe(loadError, doing: "load the 3D model"))
                        .multilineTextAlignment(.center)
                    Button("Try again") { refresh += 1 }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("heads.retry")
                }
                .frame(maxWidth: .infinity)
                .padding(24)
            } else {
                ProgressView("Loading the 3D model")
                    .frame(maxWidth: .infinity, minHeight: 160)
            }
        }
        .task(id: refresh) { await poll() }
        .confirmAction("Delete this 3D model?", isPresented: $confirmDelete, actionTitle: "Delete",
                       message: status?.state == "running"
                           ? "The heads are removed from the server and the work that is running stops. The photos are not touched, and the model can be generated again."
                           : "The heads are removed from the server. The photos are not touched, and the model can be generated again.") {
            remove()
        }
    }

    // MARK: - the parts

    @ViewBuilder
    private func content(_ s: API.Model3dOut) -> some View {
        let running = s.state == "running"
        let hasModel = !s.looks.isEmpty
        let why = HeadsWords.blocked(s)
        VStack(alignment: .leading, spacing: 16) {
            if running, let progress = s.progress {
                runningCard(s, progress, hasModel: hasModel)
            }
            if !hasModel && !running {
                HeadsCard {
                    Text("A 3D head for each look")
                        .font(.headline)
                        .accessibilityIdentifier("heads.intro")
                    Text("Builds a 3D head for each look found in \(whose) photos: ages, hair and facial hair. It runs on this server and takes a few minutes. Each head is made from one photo, so it is the model's reading of that photo and not a scan: the back of the head is a guess.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Text("\(Fmt.count(s.usablePhotos, "face photo")) \(s.usablePhotos == 1 ? "is" : "are") clear and large enough to choose from.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    failure(s, running: running)
                    if let why { notice(why) }
                    HeadsOptionSliders(value: $options, disabled: !s.available, reliefInstalled: s.reliefInstalled)
                    generateButton(why: why)
                }
            }
            if hasModel {
                failure(s, running: running)
                if !running && HeadsWords.beforePhotoTextures(s) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("These heads were made before photo textures.")
                        Button("Generate again to lay the photo over the face") { openAgain(s, jump: true) }
                            .accessibilityIdentifier("heads.againFromNote")
                    }
                    .font(.subheadline)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(.tertiarySystemFill), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
                HeadsModelView(looks: s.looks, store: store, checking: checking, onLoadError: { loadFailed() })
                    .id(s.generatedAt ?? "")
                modelCard(s, running: running, why: why)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func runningCard(_ s: API.Model3dOut, _ p: API.Model3dProgress, hasModel: Bool) -> some View {
        HeadsCard {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    (Text(s.stopping ? "Stopping" : p.step).fontWeight(.semibold)
                        + Text(!s.stopping && p.total > 0 ? ", \(p.done.formatted()) of \(p.total.formatted())" : "")
                            .foregroundColor(.secondary))
                        .font(.subheadline)
                        .accessibilityIdentifier("heads.step")
                    Text(s.stopping
                         ? "The work is being stopped. That takes a moment."
                         : "The server is working. You can leave this page; the work goes on\(hasModel ? ", and the model below stays until the new one is ready." : ".")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
                if !s.stopping {
                    Button { stop() } label: { Label("Stop", systemImage: "stop.circle") }
                        .buttonStyle(.bordered)
                        .disabled(stopping)
                        .accessibilityIdentifier("heads.stop")
                }
            }
            if p.total > 0 {
                ProgressView(value: Double(min(p.done, p.total)), total: Double(p.total))
                    .accessibilityLabel(p.step)
            } else {
                ProgressView()
                    .frame(maxWidth: .infinity)
                    .accessibilityLabel(p.step)
            }
        }
    }

    private func modelCard(_ s: API.Model3dOut, running: Bool, why: String?) -> some View {
        HeadsCard {
            Text("\(Fmt.count(s.looks.count, "look"))\(s.generatedAt != nil ? ", made \(Fmt.relative(s.generatedAt))" : ""). The heads are kept on this server only.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("heads.summary")
            HStack(spacing: 10) {
                if !running {
                    Button {
                        if again { again = false } else { openAgain(s) }
                    } label: {
                        Label("Generate again", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                    .accessibilityIdentifier("heads.again")
                }
                Button(role: .destructive) {
                    confirmDelete = true
                } label: {
                    Label("Delete this model", systemImage: "trash")
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("heads.delete")
            }
            if again && !running {
                VStack(alignment: .leading, spacing: 12) {
                    Divider()
                    Text("The photos are read again and new heads are built. The new model takes this one's place when it is ready.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    if let why { notice(why) }
                    HeadsOptionSliders(value: $options, disabled: !s.available, reliefInstalled: s.reliefInstalled)
                    HStack(spacing: 10) {
                        generateButton(why: why)
                        Button("Cancel") { again = false }
                            .accessibilityIdentifier("heads.againCancel")
                    }
                }
                .id(Self.againPanelId)
            }
        }
    }

    @ViewBuilder
    private func failure(_ s: API.Model3dOut, running: Bool) -> some View {
        if let error = s.error, !running {
            Label("About the last run: \(error)", systemImage: "exclamationmark.triangle")
                .font(.subheadline)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .accessibilityIdentifier("heads.lastRun")
        }
    }

    private func notice(_ text: String) -> some View {
        Text(text)
            .font(.subheadline)
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.tertiarySystemFill), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .accessibilityIdentifier("heads.blocked")
    }

    private func generateButton(why: String?) -> some View {
        Button {
            generate()
        } label: {
            HStack(spacing: 8) {
                if starting { ProgressView() }
                Label("Generate", systemImage: "cube")
            }
        }
        .buttonStyle(.borderedProminent)
        .disabled(why != nil || starting)
        .accessibilityIdentifier("heads.generate")
    }

    // MARK: - reading and writing

    private func poll() async {
        while !Task.isCancelled {
            await load()
            checking = false
            guard let s = status else { return }
            let seconds: UInt64
            if s.state == "running" {
                seconds = 2
            } else if s.busy {
                seconds = 5
            } else {
                return
            }
            try? await Task.sleep(nanoseconds: seconds * 1_000_000_000)
        }
    }

    private func load() async {
        do {
            let next = try await Cloud.api().getModel3d(personId: personId)
            status = next
            loadError = nil
        } catch {
            if error is CancellationError { return }
            if let u = error as? URLError, u.code == .cancelled { return }
            if status == nil { loadError = error }
        }
    }

    /// Open the panel with the options as this model was made; what a model from before an option existed does not
    /// say starts at its default. From the line above the head (`jump`) the panel is brought into view; when it is
    /// open already, it is only brought into view and keeps the options set in it.
    private func openAgain(_ s: API.Model3dOut, jump: Bool = false) {
        if !again {
            options = HeadsOptions.complete(s.options)
            again = true
        }
        guard jump, let reveal else { return }
        Task { @MainActor in
            // the panel is drawn on the next pass
            try? await Task.sleep(nanoseconds: 100_000_000)
            withAnimation { reveal(Self.againPanelId) }
        }
    }

    /// A head or a look's picture did not load: it may belong to a look the server stopped listing, so the status is
    /// read again and that look drops out. Pictures failing together ask once.
    private func loadFailed() {
        let now = Date()
        guard now.timeIntervalSince(askedAt) >= 1 else { return }
        askedAt = now
        checking = true
        refresh += 1
    }

    private func generate() {
        guard !starting else { return }
        starting = true
        let request = options.body
        Task {
            defer { starting = false }
            do {
                _ = try await UndoCenter.shared.perform("Started making the 3D heads", toast: false) {
                    try await Cloud.api().startModel3d(personId: personId, body: request)
                }
                again = false
                // show the job at once; the next read brings the real step
                if var s = status {
                    s.state = "running"
                    s.error = nil
                    if s.progress == nil { s.progress = API.Model3dProgress(step: "Starting", done: 0, total: 0) }
                    status = s
                }
            } catch {
                Toaster.shared.error(HeadsWords.startError(error))
            }
            refresh += 1
        }
    }

    private func stop() {
        guard !stopping else { return }
        stopping = true
        Task {
            defer { stopping = false }
            do {
                _ = try await UndoCenter.shared.perform("Stopped making the 3D heads", toast: false) {
                    try await Cloud.api().stopModel3d(personId: personId)
                }
                // the job ends within about a second; until the next read says so, the card says it is stopping
                if var s = status, s.state == "running" {
                    s.stopping = true
                    status = s
                }
                refresh += 1
            } catch {
                Toaster.shared.error(error, doing: "stop")
            }
        }
    }

    private func remove() {
        Task {
            do {
                _ = try await UndoCenter.shared.perform("Deleted the 3D model", toast: false) {
                    try await Cloud.api().deleteModel3d(personId: personId)
                }
                again = false
                // the server stops listing and serving the model at once; a job that was running is on its way out
                if var s = status {
                    s.stopping = s.state == "running"
                    s.looks = []
                    s.generatedAt = nil
                    s.options = nil
                    s.error = nil
                    status = s
                }
                Toaster.shared.ok("The 3D model is deleted.")
                refresh += 1
            } catch {
                Toaster.shared.error(error, doing: "delete")
            }
        }
    }
}

/// The head, its looks and its controls (the website's ModelView).
struct HeadsModelView: View {
    let looks: [API.Look3dOut]
    let store: HeadsStore
    /// The status is being read again after a failure: a failed head says "Checking the model" meanwhile.
    var checking = false
    let onLoadError: () -> Void

    @Environment(AppRouter.self) private var router
    @State private var lookKey: String?
    @State private var view = HeadsView.start
    @State private var light = HeadsLight.start
    @State private var colour = true
    @State private var photo = true
    @State private var spin = false

    var body: some View {
        let axes = HeadsLooks.axes(looks)
        if let active = looks.first(where: { $0.key == lookKey }) ?? HeadsLooks.first(looks) {
            let ready = store.prepared(for: active.meshUrl)
            // until the head is read the server's word counts; then whether its photo really draws
            let showsPhoto = active.textured && (ready == nil || ready?.photo != nil)
            VStack(alignment: .leading, spacing: 18) {
                stage(active, ready: ready)
                caption(active, showsPhoto: showsPhoto)
                strip(active)
                lookSliders(axes, active)
                viewSliders
                lightSliders
                switches(showsPhoto: showsPhoto)
            }
            .task(id: active.meshUrl) { store.want(active) }
            .onChange(of: store.states[active.meshUrl]) { _, state in
                if case .failed? = state { onLoadError() }
            }
        }
    }

    private func stage(_ active: API.Look3dOut, ready: HeadsPrepared?) -> some View {
        ZStack {
            HeadsStage(prepared: ready, lookKey: active.key, view: $view, light: light, colour: colour, photo: photo,
                       spin: spin, evenLight: looks.contains(where: { $0.textured }), label: active.label)
            switch store.states[active.meshUrl] {
            case .failed? where checking:
                ProgressView("Checking the model")
                    .padding(12)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .accessibilityIdentifier("heads.checking")
            case .failed(let message)?:
                VStack(spacing: 10) {
                    Text(message)
                        .font(.subheadline)
                        .multilineTextAlignment(.center)
                    Button("Try again") { store.retry(active) }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("heads.loadRetry")
                }
                .padding(16)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .padding(24)
            case .ready?:
                EmptyView()
            default:
                ProgressView("Loading the head")
                    .padding(12)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .accessibilityIdentifier("heads.loading")
            }
        }
        .aspectRatio(1, contentMode: .fit)
        .frame(maxWidth: 560)
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .frame(maxWidth: .infinity)
    }

    private func caption(_ active: API.Look3dOut, showsPhoto: Bool) -> some View {
        let taken = Fmt.date(active.takenAt)
        let shows = active.photos == 1 ? "shows" : "show"
        return VStack(alignment: .leading, spacing: 6) {
            (Text("\(active.label). ").fontWeight(.semibold)
                + Text("Made from one photo\(taken.isEmpty ? "" : " taken on \(taken)"); \(Fmt.count(active.photos, "photo")) \(shows) this look."))
                .font(.subheadline)
                .accessibilityIdentifier("heads.caption")
            Button("Open the photo") { router.open(.item(uid: active.itemUid)) }
                .font(.subheadline.weight(.medium))
                .accessibilityIdentifier("heads.openPhoto")
            if showsPhoto {
                Text("From the front this is the photo; past about 50 degrees the sides and back are filled in.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            if active.relief {
                Text("The nose, lips and chin were shaped from the photo by a depth model.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            if let note = HeadsWords.sentence(active.note) {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("heads.note")
            }
        }
    }

    private func strip(_ active: API.Look3dOut) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(looks, id: \.key) { look in
                    let lines = HeadsLooks.lines(look)
                    let on = look.key == active.key
                    Button {
                        lookKey = look.key
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            RemoteImage(ref: look.pictureUrl, maxPixel: 256, onFailure: onLoadError)
                                .frame(width: 88, height: 88)
                                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                            Text(lines.0)
                                .font(.caption.weight(.medium))
                                .lineLimit(1)
                            Text(lines.1.isEmpty ? " " : lines.1)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        .frame(width: 88)
                        .padding(4)
                        .background(on ? Color.accentColor.opacity(0.15) : Color.clear,
                                    in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .stroke(on ? Color.accentColor : Color.clear, lineWidth: 2))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(look.label)
                    .accessibilityAddTraits(on ? .isSelected : [])
                    .accessibilityIdentifier("heads.look.\(look.key)")
                }
            }
            .padding(.vertical, 2)
        }
        .accessibilityLabel("Looks")
    }

    private func lookSliders(_ axes: [HeadsLookAxis], _ active: API.Look3dOut) -> some View {
        HeadsGroup(title: "Look") {
            ForEach(axes.filter { !$0.stops.isEmpty }, id: \.key) { axis in
                let many = axis.stops.count > 1
                HeadsSlider(label: axis.label,
                            value: Binding(get: { Double(many ? max(0, HeadsLooks.stop(of: active.key, on: axis)) : 1) },
                                           set: { stop in
                                               lookKey = HeadsLooks.pick(axes, from: active.key, moved: axis.key,
                                                                         to: Int(stop.rounded()))
                                           }),
                            range: 0...Double(many ? axis.stops.count - 1 : 2), step: 1,
                            text: HeadsLooks.text(axis, for: active),
                            identifier: "heads.axis.\(axis.key.rawValue)")
                    .disabled(!many)
            }
        }
    }

    private var viewSliders: some View {
        HeadsGroup(title: "View") {
            HeadsSlider(label: "Turn",
                        value: Binding(get: { view.turn.rounded() }, set: { turn in
                            spin = false
                            view.turn = turn
                        }),
                        range: HeadsView.turnRange, step: 1,
                        text: HeadsWords.degrees(view.turn, zero: "facing you", half: "from behind",
                                                 positive: "to the right", negative: "to the left"),
                        identifier: "heads.turn")
            HeadsSlider(label: "Tilt", value: Binding(get: { view.tilt.rounded() }, set: { view.tilt = $0 }),
                        range: HeadsView.tiltRange, step: 1,
                        text: HeadsWords.degrees(view.tilt, zero: "level", half: "level", positive: "from above",
                                                 negative: "from below"),
                        identifier: "heads.tilt")
            HeadsSlider(label: "Zoom", value: Binding(get: { (view.zoom * 20).rounded() / 20 }, set: { view.zoom = $0 }),
                        range: HeadsView.zoomRange, step: 0.05, text: HeadsWords.percent(view.zoom),
                        identifier: "heads.zoom")
        }
    }

    private var lightSliders: some View {
        HeadsGroup(title: "Light") {
            HeadsSlider(label: "Light direction", value: $light.direction, range: HeadsLight.directionRange, step: 5,
                        text: HeadsWords.degrees(light.direction, zero: "from the front", half: "from behind",
                                                 positive: "to the right", negative: "to the left"),
                        identifier: "heads.lightDirection")
            HeadsSlider(label: "Light height", value: $light.height, range: HeadsLight.heightRange, step: 5,
                        text: HeadsWords.degrees(light.height, zero: "level", half: "level", positive: "above",
                                                 negative: "below"),
                        identifier: "heads.lightHeight")
            HeadsSlider(label: "Brightness", value: $light.brightness, range: HeadsLight.brightnessRange, step: 0.05,
                        text: HeadsWords.percent(light.brightness), identifier: "heads.brightness")
        }
    }

    private func switches(showsPhoto: Bool) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Toggle("Colour", isOn: $colour)
                .accessibilityIdentifier("heads.colour")
            if showsPhoto {
                // without colours there is no photo either: the switch waits, and comes back as it was
                Toggle("Photo", isOn: Binding(get: { colour && photo }, set: { photo = $0 }))
                    .disabled(!colour)
                    .accessibilityIdentifier("heads.photo")
            }
            Toggle("Spin slowly", isOn: $spin)
                .accessibilityIdentifier("heads.spin")
            Text("Drag the head to turn it; pinch to zoom. With Colour off you see the bare shape.\(showsPhoto ? " With Photo off you see the colours under the photo." : "")")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button {
                spin = false
                colour = true
                photo = true
                light = HeadsLight.start
                view = HeadsView.start
            } label: {
                Label("Reset view", systemImage: "arrow.counterclockwise")
            }
            .buttonStyle(.bordered)
            .accessibilityIdentifier("heads.reset")
        }
    }
}

/// The choices of a generation, as sliders (the website's OptionSliders).
struct HeadsOptionSliders: View {
    @Binding var value: HeadsOptions
    var disabled = false
    /// False when the server says the depth models of Face relief are not installed.
    var reliefInstalled: Bool?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HeadsSlider(label: "Looks", value: whole(\.maxLooks), range: 1...12, step: 1,
                        text: "at most \(value.maxLooks)", identifier: "heads.option.looks")
            HeadsSlider(label: "Detail", value: whole(\.detail), range: 1...3, step: 1,
                        text: HeadsOptions.word(HeadsOptions.detailWords, value.detail, offset: 1),
                        identifier: "heads.option.detail")
            HeadsSlider(label: "Framing", value: $value.framing, range: 1.8...3.2, step: 0.1,
                        text: String(format: "%.1f times the face", value.framing), identifier: "heads.option.framing")
            HeadsSlider(label: "Smoothing", value: whole(\.smoothing), range: 0...3, step: 1,
                        text: HeadsOptions.word(HeadsOptions.smoothingWords, value.smoothing),
                        identifier: "heads.option.smoothing")
            HeadsSlider(label: "Photo texture", value: whole(\.texture), range: 0...3, step: 1,
                        text: HeadsOptions.word(HeadsOptions.textureWords, value.texture),
                        identifier: "heads.option.texture")
            Toggle("Face relief", isOn: Binding(get: { value.relief > 0 }, set: { value.relief = $0 ? 1 : 0 }))
                .font(.subheadline.weight(.medium))
                .accessibilityIdentifier("heads.option.relief")
            Text("More looks, finer detail and a larger texture take longer and make larger files. Framing is how much of the photo around the face is used: a wider frame keeps more of the hair and the shoulders. Smoothing evens out the surface. Face relief uses a depth model made for pictures of people to shape the nose, lips and chin, when it is installed on the server.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if reliefInstalled == false && value.relief > 0 {
                Text("The depth model is not installed on this server, so heads are made without Face relief. Running scripts/setup_3d.sh on the server once more fetches it (0.9 GB).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(value.texture > 0
                 ? "The photo is laid over the front of the head, and the sides and back, which one photo does not show, are filled in. The texture is the photo at its own resolution: full keeps all of it, standard and fine shrink a large photo."
                 : "With Photo texture off the head has plain colours only, without the photo laid over its front.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(disabled)
    }

    private func whole(_ key: WritableKeyPath<HeadsOptions, Int>) -> Binding<Double> {
        Binding(get: { Double(value[keyPath: key]) }, set: { value[keyPath: key] = Int($0.rounded()) })
    }
}

/// A slider with its name and its value in words above it, read by VoiceOver as those words.
struct HeadsSlider: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let text: String
    let identifier: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline) {
                Text(label)
                    .font(.subheadline.weight(.medium))
                Spacer(minLength: 8)
                Text(text)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            .accessibilityHidden(true)
            Slider(value: $value, in: range, step: step) {
                Text(label)
            }
            .accessibilityValue(text)
            .accessibilityIdentifier(identifier)
        }
    }
}

/// A titled group of controls.
struct HeadsGroup<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .accessibilityAddTraits(.isHeader)
            content()
        }
    }
}

/// A rounded card around a part of the tab.
struct HeadsCard<Content: View>: View {
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}
