// Settings > Sessions: every signed-in browser, this one marked. Others can be revoked one by one;
// revoking this one is the same as signing out. Below them, the Devices section lists the device
// tokens the iOS app signs in with: adding one shows its token once (and needs the vault open, the
// server answers 403 vault_locked otherwise, like adding a passkey), ending one signs that app out.
import { useId, useRef, useState, type FormEvent } from "react";
import { useNavigate } from "react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  Copy,
  Laptop,
  Plus,
  Smartphone,
  TabletSmartphone,
} from "lucide-react";
import { api } from "../../../lib/api";
import {
  SESSIONS_KEY,
  authErrorMessage,
  isVaultLocked,
  passkeysSupported,
  useAuth,
} from "../../../lib/auth";
import { formatDate, formatRelative } from "../../../lib/format";
import type { AuthState, DeviceCreated, SessionOut } from "../../../lib/types";
import {
  Badge,
  Button,
  ConfirmModal,
  ErrorState,
  Modal,
  Spinner,
} from "../../ui";
import { inputClass } from "../../admin/fields";
import { useToast } from "../../Toast";
import UnlockForm from "../../vault/UnlockForm";
import { Section, describeAgent, errorText } from "./common";

const DEVICES_KEY = ["auth", "devices"] as const;
const DEVICE_NAME_MAX = 80;

function isPhone(ua: string | null): boolean {
  return !!ua && /iPhone|Android|Mobile/.test(ua);
}

export default function SessionsTab() {
  const qc = useQueryClient();
  const toast = useToast();
  const navigate = useNavigate();
  const { logout } = useAuth();
  const q = useQuery({ queryKey: SESSIONS_KEY, queryFn: api.auth.sessions });
  const [revoking, setRevoking] = useState<SessionOut | null>(null);
  const [busy, setBusy] = useState(false);

  const confirm = async () => {
    if (!revoking) return;
    setBusy(true);
    try {
      if (revoking.current) {
        await logout();
        navigate("/login", { replace: true });
        return;
      }
      await api.auth.revokeSession(revoking.id);
      toast.success(`${describeAgent(revoking.user_agent)} was signed out.`);
      setRevoking(null);
      void qc.invalidateQueries({ queryKey: SESSIONS_KEY });
    } catch (err) {
      toast.error(errorText(err, "Could not end that session."));
    } finally {
      setBusy(false);
    }
  };

  const rows = [...(q.data ?? [])].sort(
    (a, b) =>
      Number(b.current) - Number(a.current) ||
      b.last_seen_at.localeCompare(a.last_seen_at),
  );

  return (
    <div className="flex flex-col gap-4">
      <Section
        title="Signed-in sessions"
        description="Sessions last 30 days from their last use. End any you do not recognise."
      >
        {q.isPending ? (
          <div className="flex justify-center py-6">
            <Spinner label="Loading sessions" />
          </div>
        ) : q.isError ? (
          <ErrorState error={q.error} onRetry={() => void q.refetch()} />
        ) : (
          <ul className="divide-y divide-line">
            {rows.map((s) => {
              const Icon = isPhone(s.user_agent) ? Smartphone : Laptop;
              return (
                <li key={s.id} className="flex items-center gap-3 py-3">
                  <span className="flex size-10 shrink-0 items-center justify-center rounded-full bg-surface-2 text-ink-soft">
                    <Icon className="size-5" aria-hidden />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="flex flex-wrap items-center gap-2 text-sm font-semibold">
                      {describeAgent(s.user_agent)}
                      {s.current && <Badge tone="ok">This device</Badge>}
                    </p>
                    <p className="text-xs text-ink-soft">
                      {s.ip ? `${s.ip}, ` : ""}signed in{" "}
                      {formatDate(s.created_at)}, last active{" "}
                      {formatDate(s.last_seen_at, true)}
                    </p>
                  </div>
                  <Button
                    variant={s.current ? "secondary" : "danger"}
                    size="sm"
                    className="h-10"
                    onClick={() => setRevoking(s)}
                  >
                    {s.current ? "Sign out" : "Revoke"}
                  </Button>
                </li>
              );
            })}
          </ul>
        )}
        <ConfirmModal
          open={!!revoking}
          title={
            revoking?.current ? "Sign out of this device" : "Revoke session"
          }
          body={
            revoking?.current
              ? "You will need the password or a passkey to sign in again."
              : `${describeAgent(revoking?.user_agent)} will be signed out at once.`
          }
          confirmLabel={revoking?.current ? "Sign out" : "Revoke"}
          danger={!revoking?.current}
          loading={busy}
          onConfirm={() => void confirm()}
          onClose={() => setRevoking(null)}
        />
      </Section>
      <DevicesSection />
    </div>
  );
}

function DevicesSection() {
  const qc = useQueryClient();
  const toast = useToast();
  const { state, setState } = useAuth();
  const q = useQuery({ queryKey: DEVICES_KEY, queryFn: api.auth.devices });
  const nameId = useId();
  const [name, setName] = useState("My iPhone");
  const [adding, setAdding] = useState(false);
  const [created, setCreated] = useState<DeviceCreated | null>(null);
  /** The name of an add that was refused because the vault is closed, run again after the unlock. */
  const [pending, setPending] = useState<string | null>(null);
  const [ending, setEnding] = useState<SessionOut | null>(null);
  const [endBusy, setEndBusy] = useState(false);

  const refresh = () => void qc.invalidateQueries({ queryKey: DEVICES_KEY });

  /** `retried` is true for the one automatic retry after an unlock: a second vault_locked is shown, not asked again. */
  const runAdd = async (clean: string, retried: boolean) => {
    setAdding(true);
    try {
      const d = await api.auth.createDevice(clean.slice(0, DEVICE_NAME_MAX));
      setCreated(d);
      refresh();
    } catch (err) {
      if (!retried && isVaultLocked(err)) setPending(clean);
      else
        toast.error(
          isVaultLocked(err)
            ? authErrorMessage(err)
            : errorText(err, "Could not add the device."),
        );
    } finally {
      setAdding(false);
    }
  };

  const add = (e: FormEvent) => {
    e.preventDefault();
    const clean = name.replace(/\s+/g, " ").trim();
    if (!clean || adding) return;
    void runAdd(clean, false);
  };

  const unlocked = (s: AuthState) => {
    setState(s);
    const again = pending;
    setPending(null);
    if (again) void runAdd(again, true);
  };

  const cancelUnlock = () => {
    if (pending)
      toast.error("The device was not added: the vault stayed locked.");
    setPending(null);
  };

  const end = async () => {
    if (!ending) return;
    setEndBusy(true);
    try {
      await api.auth.revokeDevice(ending.id);
      toast.success(`${deviceLabel(ending)} was signed out.`);
      setEnding(null);
      refresh();
    } catch (err) {
      toast.error(errorText(err, "Could not end that device."));
    } finally {
      setEndBusy(false);
    }
  };

  const rows = [...(q.data ?? [])].sort((a, b) =>
    b.last_seen_at.localeCompare(a.last_seen_at),
  );

  return (
    <Section
      title="Devices"
      description="A device token lets the iOS app sign in and sync your photos. It keeps working for a year from its last use, and you can end it here at any time."
    >
      <form onSubmit={add} className="flex max-w-lg flex-col gap-2">
        <label htmlFor={nameId} className="text-sm font-medium">
          Device name
        </label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            id={nameId}
            value={name}
            maxLength={DEVICE_NAME_MAX}
            onChange={(e) => setName(e.target.value)}
            placeholder="For example, My iPhone"
            className={inputClass}
          />
          <Button
            type="submit"
            loading={adding}
            disabled={!name.trim()}
            className="shrink-0"
          >
            {!adding && <Plus className="size-4" aria-hidden />}
            Add device
          </Button>
        </div>
      </form>

      <div className="mt-4">
        {q.isPending ? (
          <div className="flex justify-center py-6">
            <Spinner label="Loading devices" />
          </div>
        ) : q.isError ? (
          <ErrorState error={q.error} onRetry={() => void q.refetch()} />
        ) : rows.length === 0 ? (
          <p className="text-sm text-ink-soft">No devices yet.</p>
        ) : (
          <ul className="divide-y divide-line">
            {rows.map((d) => (
              <li key={d.id} className="flex items-center gap-3 py-3">
                <span className="flex size-10 shrink-0 items-center justify-center rounded-full bg-surface-2 text-ink-soft">
                  <TabletSmartphone className="size-5" aria-hidden />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold">
                    {deviceLabel(d)}
                  </p>
                  <p className="text-xs text-ink-soft">
                    Added {formatDate(d.created_at)}, last used{" "}
                    <span title={formatDate(d.last_seen_at, true)}>
                      {formatRelative(d.last_seen_at)}
                    </span>
                  </p>
                </div>
                <Button
                  variant="danger"
                  size="sm"
                  className="h-10 shrink-0"
                  onClick={() => setEnding(d)}
                >
                  End
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <TokenModal device={created} onClose={() => setCreated(null)} />

      <ConfirmModal
        open={!!ending}
        title="End device"
        body={
          <>
            <span className="font-semibold text-ink">
              {ending ? deviceLabel(ending) : ""}
            </span>{" "}
            will be signed out at once. The app on it stops syncing until you
            add it again with a new token.
          </>
        }
        confirmLabel="End"
        danger
        loading={endBusy}
        onConfirm={() => void end()}
        onClose={() => setEnding(null)}
      />

      <Modal
        open={pending !== null}
        onClose={cancelUnlock}
        title="Confirm it is you"
      >
        <UnlockForm
          compact
          hasPasskeys={passkeysSupported() && (state?.passkeys ?? 0) > 0}
          onUnlocked={unlocked}
          description="To add a device, unlock the vault with your password or a passkey."
        />
      </Modal>
    </Section>
  );
}

function deviceLabel(d: SessionOut): string {
  return d.device_name || describeAgent(d.user_agent);
}

/** Shows a new device token once, with a Copy button. */
function TokenModal({
  device,
  onClose,
}: {
  device: DeviceCreated | null;
  onClose: () => void;
}) {
  const toast = useToast();
  const tokenRef = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);

  const selectToken = () => {
    const el = tokenRef.current;
    const sel = window.getSelection();
    if (!el || !sel) return;
    const range = document.createRange();
    range.selectNodeContents(el);
    sel.removeAllRanges();
    sel.addRange(range);
  };

  const copy = async () => {
    if (!device) return;
    try {
      if (!navigator.clipboard) throw new Error("no clipboard");
      await navigator.clipboard.writeText(device.token);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      selectToken();
      toast.error("Could not copy. The token is selected: copy it by hand.");
    }
  };

  const close = () => {
    setCopied(false);
    onClose();
  };

  return (
    <Modal
      open={!!device}
      onClose={close}
      title="Device token"
      footer={<Button onClick={close}>Done</Button>}
    >
      <div className="flex flex-col gap-3">
        <p className="text-sm text-ink-soft">
          Enter this token in the iOS app on{" "}
          <span className="font-semibold text-ink">{device?.name}</span>. It
          will not be shown again.
        </p>
        <pre
          ref={tokenRef}
          onClick={selectToken}
          className="rounded-xl bg-surface-2 px-3 py-2 font-mono text-sm break-all whitespace-pre-wrap select-all"
        >
          {device?.token}
        </pre>
        <div>
          <Button
            variant="secondary"
            size="sm"
            className="h-10"
            onClick={() => void copy()}
          >
            {copied ? (
              <Check className="size-4" aria-hidden />
            ) : (
              <Copy className="size-4" aria-hidden />
            )}
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
