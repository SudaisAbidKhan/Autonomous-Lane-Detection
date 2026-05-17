// ═══════════════════════════════════════════════════════════
//  Header.jsx  –  Top navigation bar
// ═══════════════════════════════════════════════════════════

import { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';

export default function Header({ serverStatus }) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const isOnline = serverStatus === 'online';

  return (
    <header style={{
      position: 'sticky', top: 0, zIndex: 100,
      background: scrolled ? 'rgba(10,10,8,0.95)' : 'transparent',
      backdropFilter: scrolled ? 'blur(20px)' : 'none',
      borderBottom: scrolled ? '1px solid var(--border)' : '1px solid transparent',
      transition: 'all 0.3s ease',
      padding: '0 2rem',
    }}>
      <nav style={{
        maxWidth: 1200, margin: '0 auto',
        height: 60,
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between',
      }}>

        {/* ── Logo ── */}
        <NavLink to="/" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 34, height: 34,
            background: 'var(--accent)',
            borderRadius: 8,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 18px rgba(245,158,11,0.35)',
            flexShrink: 0,
          }}>
            <RoadIcon />
          </div>
          <div>
            <div style={{
              fontFamily: 'var(--font-heading)',
              fontWeight: 800, fontSize: 16,
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)', lineHeight: 1.1,
            }}>LaneSeg</div>
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: 9,
              color: 'var(--text-muted)',
              letterSpacing: '0.08em', textTransform: 'uppercase',
            }}>CNN Lane Detection</div>
          </div>
        </NavLink>

        {/* ── Nav links + status ── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {[
            { to: '/',      label: 'Detect',      end: true  },
            { to: '/about', label: 'About Model',  end: false },
          ].map(({ to, label, end }) => (
            <NavLink key={to} to={to} end={end} style={({ isActive }) => ({
              padding: '6px 14px', borderRadius: 8,
              fontFamily: 'var(--font-body)', fontSize: 13, fontWeight: 500,
              color: isActive ? 'var(--accent)' : 'var(--text-secondary)',
              background: isActive ? 'var(--accent-glow)' : 'transparent',
              border: `1px solid ${isActive ? 'var(--accent-border)' : 'transparent'}`,
              transition: 'all var(--transition)',
            })}>
              {label}
            </NavLink>
          ))}

          {/* Server status dot */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '5px 12px', borderRadius: 100,
            background: isOnline ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)',
            border: `1px solid ${isOnline ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}`,
            marginLeft: 4,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: isOnline ? 'var(--green)' : 'var(--red)',
              display: 'inline-block',
              animation: isOnline ? 'pulse-amber 2s infinite' : 'none',
            }} />
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 10.5,
              color: isOnline ? 'var(--green)' : 'var(--red)',
            }}>
              {serverStatus === 'checking' ? 'connecting' : serverStatus}
            </span>
          </div>
        </div>
      </nav>
    </header>
  );
}

function RoadIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
         stroke="#0a0a08" strokeWidth="2.2" strokeLinecap="round">
      <path d="M3 17l3-12h12l3 12"/>
      <line x1="12" y1="5" x2="12" y2="17" strokeDasharray="2 2"/>
    </svg>
  );
}