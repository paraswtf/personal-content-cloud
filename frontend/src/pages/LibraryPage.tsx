// Library (/): every photo, video and document outside the vault, with filters and search.
import Library from '../components/library/Library'
import { usePageTitle } from '../lib/hooks'

export default function LibraryPage() {
  usePageTitle('Library')
  return <Library />
}
