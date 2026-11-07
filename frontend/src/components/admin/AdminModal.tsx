// Modal and ConfirmModal wrappers with a stable onClose.
// The shared Modal re-runs its focus effect whenever onClose changes identity, which would pull
// focus out of inputs on every parent render. These wrappers keep the identity fixed.
import type { ComponentProps } from 'react'
import { ConfirmModal, Modal } from '../ui'
import { useStable } from './adminHooks'

export function AdminModal({ onClose, ...rest }: ComponentProps<typeof Modal>) {
  const stable = useStable(onClose)
  return <Modal {...rest} onClose={stable} />
}

export function Confirm({ onClose, ...rest }: ComponentProps<typeof ConfirmModal>) {
  const stable = useStable(onClose)
  return <ConfirmModal {...rest} onClose={stable} />
}
