// Fallback for unknown paths inside the app shell.
import { Link } from 'react-router'
import { Compass, Images, Upload } from 'lucide-react'
import { EmptyState, buttonClasses } from '../components/ui'
import { usePageTitle } from '../components/shell/title'

export default function NotFoundPage() {
  usePageTitle('Page not found')
  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <EmptyState
        icon={<Compass className="size-10" aria-hidden />}
        title="This page does not exist"
        action={
          <div className="mt-2 flex flex-wrap justify-center gap-2">
            <Link to="/" className={buttonClasses('primary', 'md')}>
              <Images className="size-4" aria-hidden />
              Open the library
            </Link>
            <Link to="/upload" className={buttonClasses('secondary', 'md')}>
              <Upload className="size-4" aria-hidden />
              Upload
            </Link>
          </div>
        }
      >
        The address may be mistyped, or the item may have been deleted.
      </EmptyState>
    </div>
  )
}
