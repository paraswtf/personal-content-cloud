// Routes (docs/ARCHITECTURE.md section 14). The app shell replaces this file.
import { lazy, Suspense } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router'

const SetupPage = lazy(() => import('./pages/SetupPage'))
const LoginPage = lazy(() => import('./pages/LoginPage'))
const LibraryPage = lazy(() => import('./pages/LibraryPage'))
const PeoplePage = lazy(() => import('./pages/PeoplePage'))
const PersonPage = lazy(() => import('./pages/PersonPage'))
const DocumentsPage = lazy(() => import('./pages/DocumentsPage'))
const ReviewPage = lazy(() => import('./pages/ReviewPage'))
const VaultPage = lazy(() => import('./pages/VaultPage'))
const UploadPage = lazy(() => import('./pages/UploadPage'))
const TrashPage = lazy(() => import('./pages/TrashPage'))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))
const MediaPage = lazy(() => import('./pages/MediaPage'))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'))

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={null}>
        <Routes>
          <Route path="/setup" element={<SetupPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<LibraryPage />} />
          <Route path="/people" element={<PeoplePage />} />
          <Route path="/people/:id" element={<PersonPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/review" element={<ReviewPage />} />
          <Route path="/vault" element={<VaultPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/trash" element={<TrashPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/m/:uid" element={<MediaPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  )
}
