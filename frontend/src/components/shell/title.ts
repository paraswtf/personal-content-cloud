// Document title for shell-owned pages: "<page> | <site title>".
import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { AuthState } from '../../lib/types'
import { AUTH_KEY } from '../../lib/auth'

export const DEFAULT_SITE_TITLE = 'Personal Content Cloud'

export function usePageTitle(page?: string | null): void {
  const qc = useQueryClient()
  const site = qc.getQueryData<AuthState>(AUTH_KEY)?.site_title || DEFAULT_SITE_TITLE
  useEffect(() => {
    document.title = page ? `${page} | ${site}` : site
  }, [page, site])
}
