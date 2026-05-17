// Loader.jsx – Animated inference spinner

import { useState, useEffect } from 'react';

const MESSAGES = [
  'Preprocessing driving image…',
  'Running ResNet34 encoder…',
  'Applying skip connections…',
  'Decoding lane mask…',
  'Computing pixel probabilities…',
  'Generating overlay…',
  'Almost done…',
];

export function SpinnerInline({ size = 16, color = 'var(--accent)' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         style={{ animation: 'spin 0.8s linear infinite', flexShrink: 0 }}>
      <circle cx="12" cy="12" r="10" stroke={color} strokeWidth="2.5"
              strokeDasharray="40 20" />
    </svg>
  );
}

export default function Loader() {
  const [msgIndex, setMsgIndex] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setMsgIndex(i => (i + 1) % MESSAGES.length), 1800);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      gap: 28, padding: '52px 24px',
    }}>
      {/* Animated road icon */}
      <div style={{ position: 'relative', width: 88, height: 88 }}>
        {/* Outer ring */}
        <div style={{
          position: 'absolute', inset: -8,
          borderRadius: '50%',
          border: '1px solid var(--accent-border)',
          animation: 'pulse-amber 2s ease-in-out infinite',
        }} />
        {/* Spinning arc */}
        <svg width="88" height="88" viewBox="0 0 88 88" style={{ position: 'absolute', top: 0, left: 0 }}>
          <circle cx="44" cy="44" r="38"
            fill="none" stroke="var(--bg-elevated)" strokeWidth="3" />
          <circle cx="44" cy="44" r="38"
            fill="none" stroke="var(--accent)" strokeWidth="3"
            strokeDasharray="60 180" strokeLinecap="round"
            style={{ transformOrigin: '44px 44px', animation: 'spin 1.2s linear infinite' }} />
        </svg>
        {/* Center icon */}
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          borderRadius: '50%', background: 'var(--bg-elevated)',
          border: '1px solid var(--border)',
        }}>
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none"
               stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round">
            <path d="M3 17l3-12h12l3 12"/>
            <line x1="12" y1="5" x2="12" y2="17" strokeDasharray="2 2"/>
          </svg>
        </div>
      </div>

      {/* Message */}
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontFamily: 'var(--font-heading)', fontSize: 16, fontWeight: 700,
          color: 'var(--text-primary)', marginBottom: 8,
        }}>
          Detecting Lanes
        </div>
        <div key={msgIndex} className="fade-in" style={{
          fontFamily: 'var(--font-mono)', fontSize: 12,
          color: 'var(--text-muted)',
        }}>
          {MESSAGES[msgIndex]}
        </div>
      </div>

      {/* Progress dots */}
      <div style={{ display: 'flex', gap: 6 }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            width: 6, height: 6, borderRadius: '50%',
            background: 'var(--accent)',
            animation: `pulse-amber 1.2s ease-in-out ${i * 0.2}s infinite`,
          }} />
        ))}
      </div>
    </div>
  );
}