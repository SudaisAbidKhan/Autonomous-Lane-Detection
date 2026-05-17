// MetricsCard.jsx – Lane coverage, Dice, IoU, Accuracy tiles

function grade(v) {
  if (v === null || v === undefined) return null;
  if (v >= 0.85) return { text: 'Excellent', color: '#22c55e' };
  if (v >= 0.70) return { text: 'Good',      color: '#f59e0b' };
  if (v >= 0.55) return { text: 'Fair',      color: '#60a5fa' };
  return             { text: 'Poor',      color: '#ef4444' };
}

function Tile({ label, value, color, suffix = '', desc }) {
  const g   = suffix ? null : grade(value);
  const val = value === null || value === undefined
    ? '—'
    : suffix
    ? `${value}${suffix}`
    : value.toFixed(4);

  const pct = suffix === '%'
    ? Math.min(Math.max(parseFloat(value) || 0, 0), 100) / 100
    : Math.min(Math.max(parseFloat(value) || 0, 0), 1);

  return (
    <div style={{
      flex: 1, minWidth: 0,
      padding: '16px 14px', borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column', gap: 10,
      position: 'relative', overflow: 'hidden',
    }}>
      {/* Corner glow */}
      <div style={{
        position: 'absolute', top: 0, right: 0,
        width: 60, height: 60,
        background: `radial-gradient(circle at top right, ${color}20, transparent 70%)`,
        pointerEvents: 'none',
      }} />

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 10.5,
          color: 'var(--text-muted)', textTransform: 'uppercase',
          letterSpacing: '0.07em',
        }}>{label}</span>
        {g && (
          <span style={{
            padding: '2px 7px', borderRadius: 100,
            fontFamily: 'var(--font-mono)', fontSize: 9.5,
            background: `${g.color}15`, color: g.color,
            border: `1px solid ${g.color}30`,
          }}>{g.text}</span>
        )}
      </div>

      {/* Value */}
      <div style={{
        fontFamily: 'var(--font-heading)', fontSize: 26, fontWeight: 800,
        color: value !== null && value !== undefined ? color : 'var(--text-muted)',
        lineHeight: 1, letterSpacing: '-0.02em',
      }}>{val}</div>

      {/* Progress bar */}
      <div style={{
        height: 3, borderRadius: 100,
        background: 'var(--bg-elevated)', overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', width: `${pct * 100}%`,
          background: color, borderRadius: 100,
          transition: 'width 0.8s ease',
        }} />
      </div>

      {desc && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-muted)' }}>
          {desc}
        </div>
      )}
    </div>
  );
}

export default function MetricsCard({ metrics, laneCoverage, inferenceMs }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Section label */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{
          width: 26, height: 26, borderRadius: 7,
          background: 'var(--accent-glow)',
          border: '1px solid var(--accent-border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
               stroke="var(--accent)" strokeWidth="2" strokeLinecap="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
        </div>
        <span style={{
          fontFamily: 'var(--font-heading)', fontSize: 13.5, fontWeight: 700,
          color: 'var(--text-primary)',
        }}>Detection Metrics</span>
      </div>

      {/* Row 1 — coverage + time */}
      <div style={{ display: 'flex', gap: 10 }}>
        <Tile
          label="Lane Coverage"
          value={laneCoverage !== undefined ? parseFloat(laneCoverage).toFixed(2) : null}
          suffix="%"
          color="#f59e0b"
          desc="% of pixels classified as lane"
        />
        <Tile
          label="Inference Time"
          value={inferenceMs !== undefined ? `${parseFloat(inferenceMs).toFixed(0)}` : null}
          suffix=" ms"
          color="#60a5fa"
          desc="forward pass duration"
        />
      </div>

      {/* Row 2 — metrics (only if ground truth uploaded) */}
      {metrics ? (
        <div style={{ display: 'flex', gap: 10 }}>
          <Tile label="Dice"     value={metrics.dice}     color="#22c55e"  desc="overlap score" />
          <Tile label="IoU"      value={metrics.iou}      color="#a78bfa"  desc="intersection / union" />
          <Tile label="Accuracy" value={metrics.accuracy} color="#f59e0b"  desc="pixel accuracy" />
        </div>
      ) : (
        <div style={{
          padding: '14px 16px', borderRadius: 'var(--radius-md)',
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
          fontFamily: 'var(--font-mono)', fontSize: 11.5,
          color: 'var(--text-muted)', lineHeight: 1.6,
        }}>
          Upload a ground-truth mask to enable Dice / IoU / Accuracy metrics.
        </div>
      )}
    </div>
  );
}