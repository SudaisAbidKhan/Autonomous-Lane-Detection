// App.jsx – Root component with routing & global state

import { useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Home   from './pages/Home.jsx';
import About  from './pages/About.jsx';
import Header from './components/Header.jsx';
import { checkHealth, getModelInfo } from './services/api.js';

function PageWrapper({ children }) {
  const location = useLocation();
  return (
    <div key={location.pathname} className="slide-up" style={{ flex: 1 }}>
      {children}
    </div>
  );
}

function Footer() {
  return (
    <footer style={{ borderTop: '1px solid var(--border-subtle)', padding: '20px 2rem' }}>
      <div style={{
        maxWidth: 1200, margin: '0 auto',
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', flexWrap: 'wrap', gap: 10,
      }}>
        {['LaneSeg · KITTI Road Detection · PyTorch 2.2',
          'Deep Learning Major Assignment · 2024',
          'U-Net — Ronneberger et al., MICCAI 2015',
        ].map(t => (
          <span key={t} style={{
            fontFamily: 'var(--font-mono)', fontSize: 11,
            color: 'var(--text-muted)', letterSpacing: '0.03em',
          }}>{t}</span>
        ))}
      </div>
    </footer>
  );
}

function OfflineBanner() {
  return (
    <div style={{
      background: 'rgba(239,68,68,0.07)',
      borderBottom: '1px solid rgba(239,68,68,0.18)',
      padding: '10px 2rem', textAlign: 'center',
      fontFamily: 'var(--font-mono)', fontSize: 12.5,
      color: 'var(--red)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
    }}>
      <span>⚠</span> Flask backend is offline — start it with{' '}
      <code style={{ background: 'rgba(239,68,68,0.10)', padding: '1px 8px',
                     borderRadius: 4, fontSize: 11.5 }}>
        cd backend && python app.py
      </code>
      {' '}then refresh.
    </div>
  );
}

export default function App() {
  const [serverStatus, setServerStatus] = useState('checking');
  const [modelInfo,    setModelInfo]    = useState(null);

  const pollHealth = useCallback(async () => {
    try {
      const data = await checkHealth();
      setServerStatus(data.model_loaded ? 'online' : 'offline');
    } catch {
      setServerStatus('offline');
    }
  }, []);

  const fetchModelInfo = useCallback(async () => {
    try { setModelInfo(await getModelInfo()); } catch {}
  }, []);

  useEffect(() => {
    pollHealth(); fetchModelInfo();
    const id = setInterval(pollHealth, 10_000);
    return () => clearInterval(id);
  }, [pollHealth, fetchModelInfo]);

  return (
    <Router>
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        {serverStatus === 'offline' && <OfflineBanner />}
        <Header serverStatus={serverStatus} />
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Routes>
            <Route path="/" element={
              <PageWrapper><Home serverStatus={serverStatus} /></PageWrapper>
            } />
            <Route path="/about" element={
              <PageWrapper><About modelInfo={modelInfo} serverStatus={serverStatus} /></PageWrapper>
            } />
            <Route path="*" element={
              <PageWrapper>
                <div style={{
                  flex: 1, display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center',
                  padding: '6rem 2rem', gap: 16, textAlign: 'center',
                }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 80,
                                fontWeight: 300, color: 'var(--border)', lineHeight: 1 }}>404</div>
                  <div style={{ fontFamily: 'var(--font-heading)', fontSize: 22,
                                color: 'var(--text-secondary)' }}>Page not found</div>
                  <a href="/" style={{
                    marginTop: 8, padding: '9px 22px',
                    background: 'var(--accent-glow)', border: '1px solid var(--accent-border)',
                    borderRadius: 'var(--radius-md)', color: 'var(--accent)',
                    fontFamily: 'var(--font-body)', fontSize: 14, fontWeight: 500,
                  }}>← Back to Detection</a>
                </div>
              </PageWrapper>
            } />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}