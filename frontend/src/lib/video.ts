// Video encoder calls (docs/ARCHITECTURE.md section 27): Settings > System's Video card.
import { post } from './api'
import type { VideoStatus } from './types'

export const videoApi = {
  /** POST /api/system/video/detect: probe the encoders again (a few seconds) and return the status. */
  detect: () => post<VideoStatus>('/api/system/video/detect'),
  /** POST /api/system/video/requeue: queue the web copy again for failed videos, or for every video that needs one. */
  requeue: (scope: 'failed' | 'all') => post<{ requeued: number }>('/api/system/video/requeue', { scope }),
}
