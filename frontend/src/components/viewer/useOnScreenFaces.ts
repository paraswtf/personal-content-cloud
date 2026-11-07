// Which faces of a video are on screen at the player's current time, from their spans (the
// stretches around the sampled frames each face was detected in). Follows playback and seeking,
// and only re-renders when someone comes into or leaves the frame.
import { useEffect, useMemo, useState, type RefObject } from 'react'
import type { FaceOut } from '../../lib/types'

const EVENTS = ['timeupdate', 'seeking', 'seeked', 'loadedmetadata', 'emptied'] as const

/** Ids of the faces on screen at `t` seconds, in the order of `faces`. */
export function facesAt(faces: FaceOut[], t: number): number[] {
  return faces.filter((f) => f.spans?.some(([start, end]) => start <= t && t <= end)).map((f) => f.id)
}

/**
 * Face ids on screen in the player of `videoRef`, or null when there is no player.
 * `player` must change whenever a new <video> element is mounted (item, version or source).
 */
export function useOnScreenFaces(
  videoRef: RefObject<HTMLVideoElement | null>,
  faces: FaceOut[] | null,
  player: string | null,
): Set<number> | null {
  const [ids, setIds] = useState<number[] | null>(null)

  useEffect(() => {
    const video = videoRef.current
    if (!faces || !player || !video) {
      setIds(null)
      return
    }
    const update = () => {
      const next = facesAt(faces, video.currentTime)
      setIds((cur) => (cur && cur.length === next.length && cur.every((id, i) => id === next[i]) ? cur : next))
    }
    update()
    for (const e of EVENTS) video.addEventListener(e, update)
    return () => {
      for (const e of EVENTS) video.removeEventListener(e, update)
    }
  }, [videoRef, faces, player])

  return useMemo(() => (ids ? new Set(ids) : null), [ids])
}
