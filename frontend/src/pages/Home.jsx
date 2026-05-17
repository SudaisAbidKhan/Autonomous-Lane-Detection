// Home.jsx – Main lane detection page

import { useState, useCallback } from 'react';
import UploadPanel              from '../components/UploadPanel.jsx';
import ResultPanel              from '../components/ResultPanel.jsx';
import MetricsCard              from '../components/MetricsCard.jsx';
import Loader, { SpinnerInline } from '../components/Loader.jsx';
import { predictImage, predictWithMask } from '../services/api.js';

export default function Home({ serverStatus }) {
  const [imageFile, setImageFile] = useState(null);
  const [maskFile,  setMaskFile]  = useState(null);
  const [mode,      setMode]      = useState('idle');   // idle|loading|result|error
  const [result,    setResult]    = useState(null);
  const [errorMsg,  setErrorMsg]  = useState('');

  const isDisabled = serverStatus === 'offline' || mode === 'loading';
  const canRun     = !!imageFile && !isDisabled;

  const handleRun = useCallback(async () => {
    if (!imageFile) return;
    setMode('loading'); setResult(null); setErrorMsg('');
    try {
      const data = maskFile
        ? await predictWithMask(imageFile, maskFile)
        : await predictImage(imageFile);
      setResult(data);
      setMode('result');
    } catch (err) {
      setErrorMsg(err.message || 'Prediction failed. Is the Flask server running?');
      setMode('error');
    }
  }, [imageFile, maskFile]);

  const handleReset = () => {
    setMode('idle'); setResult(null); setErrorMsg('');
    setImageFile(null); setMaskFile(null);
  };

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '40px 2rem 64px' }}>

      {/* Hero */}
      <div className="fade-in" style={{ marginBottom: 40 }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          padding: '4px 12px', borderRadius: 100,
          background: 'var(--accent-glow)', border: '1px solid var(--accent-border)',
          marginBottom: 16,
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%',
            background: 'var(--accent)', display: 'inline-block',
            animation: 'pulse-amber 1.8s ease-in-out infinite',
          }} />
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 11,
            color: 'var(--accent)', letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}>Deep Learning · Autonomous Driving</span>
        </div>

        <h1 style={{
          fontFamily: 'var(--font-heading)',
          fontSize: 'clamp(28px, 4vw, 44px)',
          fontWeight: 800, letterSpacing: '-0.03em',
          lineHeight: 1.1, color: 'var(--text-primary)', marginBottom: 14,
        }}>
          Autonomous{' '}
          <span style={{
            background: 'linear-gradient(135deg, #f59e0b, #fcd34d)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          }}>Lane Detection</span>
        </h1>

        <p style={{
          fontFamily: 'var(--font-body)', fontSize: 15.5,
          color: 'var(--text-secondary)', maxWidth: 560, lineHeight: 1.7,
        }}>
          Upload a driving image and the U-Net model detects lane boundaries
          with a pixel-wise segmentation mask and coverage estimate.
        </p>
      </div>

      {/* Two-column layout */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: mode === 'result' ? '400px 1fr' : '460px 1fr',
        gap: 24, alignItems: 'start',
        transition: 'grid-template-columns 0.3s ease',
      }}>

        {/* LEFT */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="card" style={{ padding: 20 }}>
            <SectionLabel icon={<UploadIcon />} text="Input Driving Image" />
            <div style={{ marginTop: 14 }}>
              <UploadPanel
                onImageSelect={setImageFile}
                onMaskSelect={setMaskFile}
                disabled={isDisabled}
              />
            </div>
          </div>

          {/* Run button */}
          <button
            onClick={handleRun}
            disabled={!canRun}
            style={{
              width: '100%', padding: '14px',
              borderRadius: 'var(--radius-md)',
              border: canRun ? '1px solid var(--accent-border)' : '1px solid var(--border)',
              background: canRun
                ? 'linear-gradient(135deg, rgba(245,158,11,0.18), rgba(253,211,77,0.10))'
                : 'var(--bg-elevated)',
              color: canRun ? 'var(--accent)' : 'var(--text-muted)',
              fontFamily: 'var(--font-heading)', fontSize: 15, fontWeight: 700,
              cursor: canRun ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s', display: 'flex',
              alignItems: 'center', justifyContent: 'center', gap: 10,
              letterSpacing: '-0.01em',
              boxShadow: canRun ? '0 0 24px rgba(245,158,11,0.15)' : 'none',
            }}
            onMouseEnter={e => canRun && (
              e.currentTarget.style.boxShadow = '0 0 40px rgba(245,158,11,0.28)',
              e.currentTarget.style.transform = 'translateY(-1px)'
            )}
            onMouseLeave={e => (
              e.currentTarget.style.boxShadow = canRun ? '0 0 24px rgba(245,158,11,0.15)' : 'none',
              e.currentTarget.style.transform = 'translateY(0)'
            )}
          >
            {mode === 'loading'
              ? <><SpinnerInline size={17} color="var(--accent)" /> Detecting…</>
              : <><LaneIcon /> Detect Lanes</>}
          </button>

          {/* Offline notice */}
          {serverStatus === 'offline' && (
            <div style={{
              padding: '12px 14px', borderRadius: 'var(--radius-md)',
              background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.18)',
              fontFamily: 'var(--font-mono)', fontSize: 11.5,
              color: 'var(--red)', lineHeight: 1.6,
            }}>
              Backend offline. Start it with:<br />
              <code style={{
                display: 'inline-block', marginTop: 4,
                background: 'rgba(239,68,68,0.10)',
                padding: '2px 8px', borderRadius: 4, fontSize: 11,
              }}>cd backend && python app.py</code>
            </div>
          )}

          {/* Reset */}
          {mode === 'result' && (
            <button onClick={handleReset} style={{
              width: '100%', padding: '10px',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border)', background: 'transparent',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-body)', fontSize: 13, cursor: 'pointer',
            }}>↩ New Image</button>
          )}

          {/* Info chips */}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {['PNG · JPG · BMP', 'Max 16 MB', 'U-Net 224×224'].map(t => (
              <span key={t} style={{
                padding: '3px 10px', borderRadius: 100,
                border: '1px solid var(--border)',
                fontFamily: 'var(--font-mono)', fontSize: 10.5,
                color: 'var(--text-muted)',
              }}>{t}</span>
            ))}
          </div>
        </div>

        {/* RIGHT */}
        <div>
          {mode === 'idle' && <IdlePlaceholder />}

          {mode === 'loading' && (
            <div className="card" style={{ padding: 0 }}>
              <Loader />
            </div>
          )}

          {mode === 'error' && (
            <ErrorPanel message={errorMsg} onRetry={handleRun} />
          )}

          {mode === 'result' && result && (
            <div className="slide-up" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div className="card" style={{ padding: 18 }}>
                <SectionLabel icon={<ResultIcon />} text="Lane Segmentation Result" />
                <div style={{ marginTop: 14 }}>
                  <ResultPanel result={result} />
                </div>
              </div>
              <div className="card" style={{ padding: 18 }}>
                <MetricsCard
                  metrics={result.metrics || null}
                  laneCoverage={result.lane_coverage}
                  inferenceMs={result.inference_ms}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* How it works */}
      {mode === 'idle' && <HowItWorks />}
    </div>
  );
}

// ── Idle placeholder ─────────────────────────────────────────
function IdlePlaceholder() {
  return (
    <div className="card" style={{
      minHeight: 360, display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      padding: '48px 32px', gap: 16, textAlign: 'center',
    }}>
      <div style={{ position: 'relative', marginBottom: 8 }}>
        <div style={{
          width: 120, height: 120, borderRadius: 'var(--radius-xl)',
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          overflow: 'hidden', position: 'relative',
        }}>
          {/* Road lines */}
          {[20, 50, 80].map(y => (
            <div key={y} style={{
              position: 'absolute', left: 0, right: 0, top: y, height: 1,
              background: 'rgba(245,158,11,0.08)',
            }} />
          ))}
          <svg width="52" height="52" viewBox="0 0 24 24" fill="none"
               stroke="rgba(245,158,11,0.3)" strokeWidth="1.2" strokeLinecap="round">
            <path d="M3 17l3-12h12l3 12"/>
            <line x1="12" y1="5" x2="12" y2="17" strokeDasharray="2 2"
                  stroke="rgba(245,158,11,0.5)"/>
          </svg>
          {/* Scan line */}
          <div style={{
            position: 'absolute', left: 0, right: 0, height: 2,
            background: 'linear-gradient(90deg, transparent, var(--accent), transparent)',
            animation: 'scan-h 2.5s ease-in-out infinite',
          }} />
        </div>
        {/* Orbit ring */}
        <div style={{
          position: 'absolute', inset: -12, borderRadius: '50%',
          border: '1px dashed var(--accent-border)',
          animation: 'spin 14s linear infinite',
        }} />
        <div style={{
          position: 'absolute', top: -6, left: '50%',
          width: 8, height: 8, borderRadius: '50%',
          background: 'var(--accent)', transform: 'translateX(-50%)',
          boxShadow: '0 0 10px var(--accent)',
        }} />
      </div>
      <div style={{ fontFamily: 'var(--font-heading)', fontSize: 18, fontWeight: 700,
                    color: 'var(--text-secondary)' }}>Awaiting Driving Image</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12,
                    color: 'var(--text-muted)', maxWidth: 260, lineHeight: 1.7 }}>
        Upload a road scene image to detect lane boundaries
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'center' }}>
        {['PNG', 'JPG', 'BMP'].map(f => (
          <span key={f} style={{
            padding: '3px 10px', borderRadius: 100,
            border: '1px solid var(--border)',
            fontFamily: 'var(--font-mono)', fontSize: 10.5,
            color: 'var(--text-muted)',
          }}>{f}</span>
        ))}
      </div>
    </div>
  );
}

// ── Error panel ──────────────────────────────────────────────
function ErrorPanel({ message, onRetry }) {
  return (
    <div className="card slide-up" style={{
      padding: 28, textAlign: 'center',
      border: '1px solid rgba(239,68,68,0.2)',
      background: 'rgba(239,68,68,0.05)',
    }}>
      <div style={{ fontSize: 32, marginBottom: 12 }}>⚠️</div>
      <div style={{ fontFamily: 'var(--font-heading)', fontSize: 17, fontWeight: 700,
                    color: 'var(--red)', marginBottom: 8 }}>Detection Failed</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12,
                    color: 'var(--text-muted)', marginBottom: 20, lineHeight: 1.6 }}>
        {message}
      </div>
      <button onClick={onRetry} style={{
        padding: '9px 24px', borderRadius: 'var(--radius-md)',
        border: '1px solid rgba(239,68,68,0.3)',
        background: 'rgba(239,68,68,0.10)',
        color: 'var(--red)',
        fontFamily: 'var(--font-body)', fontSize: 13, fontWeight: 500,
        cursor: 'pointer',
      }}>Retry</button>
    </div>
  );
}

// ── How it works ─────────────────────────────────────────────
function HowItWorks() {
  const steps = [
    { n: '01', title: 'Upload Image', desc: 'Drop a road scene PNG, JPG, or BMP from the KITTI dataset or your own footage.' },
    { n: '02', title: 'U-Net Inference', desc: 'ResNet34-encoder U-Net runs a full forward pass to predict lane pixel probabilities.' },
    { n: '03', title: 'View Results', desc: 'Inspect the binary mask, green overlay, probability heatmap, and coverage metrics.' },
  ];
  return (
    <div style={{ marginTop: 60 }}>
      <div style={{ textAlign: 'center', marginBottom: 32 }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-muted)',
                      letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 10 }}>
          How it works
        </div>
        <div style={{ fontFamily: 'var(--font-heading)', fontSize: 22, fontWeight: 700,
                      color: 'var(--text-primary)' }}>Three steps to lane detection</div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        {steps.map(({ n, title, desc }) => (
          <div key={n} className="card" style={{ padding: '22px 18px' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--accent)',
                          letterSpacing: '0.06em', marginBottom: 10 }}>{n}</div>
            <div style={{ fontFamily: 'var(--font-heading)', fontSize: 15, fontWeight: 700,
                          color: 'var(--text-primary)', marginBottom: 8 }}>{title}</div>
            <div style={{ fontFamily: 'var(--font-body)', fontSize: 13,
                          color: 'var(--text-secondary)', lineHeight: 1.65 }}>{desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────
function SectionLabel({ icon, text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        width: 26, height: 26, borderRadius: 7,
        background: 'var(--accent-glow)', border: '1px solid var(--accent-border)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>{icon}</div>
      <span style={{ fontFamily: 'var(--font-heading)', fontSize: 13.5, fontWeight: 700,
                     color: 'var(--text-primary)' }}>{text}</span>
    </div>
  );
}
function UploadIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="var(--accent)" strokeWidth="2" strokeLinecap="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}
function LaneIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M3 17l3-12h12l3 12"/>
      <line x1="12" y1="5" x2="12" y2="17" strokeDasharray="2 2"/>
    </svg>
  );
}
function ResultIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
         stroke="var(--accent)" strokeWidth="2" strokeLinecap="round">
      <rect x="3" y="3" width="18" height="18" rx="2"/>
      <path d="M3 9h18M9 21V9"/>
    </svg>
  );
}