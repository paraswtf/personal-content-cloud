// Shown by AppShell when the server's API build (GET /api/auth/state api_build) differs from the one
// this site was built against (__API_BUILD__, see scripts/api-build.mjs): the two run different code,
// so pages can break on fields one side does not know, until the server is restarted and the page
// reloaded. Nothing is shown when either side has no build id.
export default function BuildBanner({ server, site }: { server: string; site: string }) {
  return (
    <div role="status" className="border-b border-line bg-surface-2 px-4 py-2 text-sm text-ink">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2">
        <span>
          This site and the server were started from different versions (site {site}, server {server}). Restart
          the server from the portal, then reload this page.
        </span>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="rounded-full border border-line bg-surface px-3 py-1 text-xs font-semibold hover:bg-bg"
        >
          Reload
        </button>
      </div>
    </div>
  )
}
