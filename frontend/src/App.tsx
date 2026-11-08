// Routes (docs/ARCHITECTURE.md section 14). /setup and /login stand alone; every other page renders
// inside AppShell, which guards the session and draws the header and navigation. Pages are lazy
// chunks; ToastProvider wraps everything so any page or portal can raise a toast. The outer
// ErrorBoundary catches a crash in the shell or the auth pages; AppShell has its own around the page.
import { lazy, Suspense } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router'
import { PageSpinner } from './components/ui'
import { ToastProvider } from './components/Toast'
import ErrorBoundary from './components/ErrorBoundary'
import AppShell from './components/shell/AppShell'

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
const AlbumsPage = lazy(() => import('./pages/AlbumsPage'))
const AlbumPage = lazy(() => import('./pages/AlbumPage'))
const MapPage = lazy(() => import('./pages/MapPage'))
const SharePage = lazy(() => import('./pages/SharePage'))

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <ErrorBoundary>
        <Suspense fallback={<PageSpinner />}>
          <Routes>
            <Route path="/setup" element={<SetupPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/s/:token" element={<SharePage />} />
            <Route element={<AppShell />}>
              <Route path="/" element={<LibraryPage />} />
              <Route path="/people" element={<PeoplePage />} />
              <Route path="/people/:id" element={<PersonPage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/albums" element={<AlbumsPage />} />
              <Route path="/albums/:id" element={<AlbumPage />} />
              <Route path="/map" element={<MapPage />} />
              <Route path="/review" element={<ReviewPage />} />
              <Route path="/vault" element={<VaultPage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/trash" element={<TrashPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/m/:uid" element={<MediaPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Route>
          </Routes>
        </Suspense>
        </ErrorBoundary>
      </ToastProvider>
    </BrowserRouter>
  )
}
