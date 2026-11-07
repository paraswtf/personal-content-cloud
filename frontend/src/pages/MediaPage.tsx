// Standalone viewer for one item (/m/:uid). `?vault=1` opens a vault item. Closing goes back when
// there is somewhere to go back to, otherwise to the library (or the vault).
import { useNavigate, useParams, useSearchParams } from 'react-router'
import Viewer from '../components/Viewer'
import { usePageTitle } from '../lib/hooks'

export default function MediaPage() {
  usePageTitle('Item')
  const { uid } = useParams()
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const vault = params.get('vault') === '1'
  const t = Number(params.get('t'))
  const close = () => {
    const idx = (window.history.state as { idx?: number } | null)?.idx ?? 0
    if (idx > 0) navigate(-1)
    else navigate(vault ? '/vault' : '/', { replace: true })
  }
  if (!uid) return null
  return (
    <div className="min-h-dvh bg-[#111]">
      <Viewer uid={uid} onClose={close} vault={vault} startAt={Number.isFinite(t) && t > 0 ? t : null} mode={params.get('doc') === '1' ? 'document' : null} />
    </div>
  )
}
