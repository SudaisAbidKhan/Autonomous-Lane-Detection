// ResultPanel.jsx – Tabbed result image viewer

import { useState } from 'react';
import { b64ToDataUri, downloadB64Image } from '../services/api.js';

const TABS = [
  { key: 'overlay',  label: 'Overlay',    color: '#f59e0b' },
  { key: 'original', label: 'Original',   color: '#60a5fa' },
  { key: 'mask',     label: 'Pred. Mask', color: '#22c55e' },
  { key: 'heatmap',  label: 'Heatmap',    color: '#a78bfa' },
  { key: 'gt',       label: 'Ground Truth', color: '#34d399' },
];

export default function ResultPanel({ result }) {
  const [active, setActive] = useState('overlay');

  if (!result) return null;

  const hasGT = !!result.gt_mask_b64;
  const imageMap = {
    original: result.original_b64,
    overlay:  result.overlay_b64,
    mask:     result.pred_mask_b64,
    heatmap:  result.prob_map_b64,
    gt:       result.gt_mask_b64,
  };

  const visibleTabs = TABS.filter(t => t.key !== 'gt' || hasGT);
  const activeTab   = visibleTabs.find(t => t.key === active) || visibleTabs[0];
  const src         = imageMap[activeTab.key];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', alignItems: 'center' }}>
        {visibleTabs.map(tab => (
          <button
            key={tab.key}
            onClick={() => setActive(tab.key)}
            style={{
              padding: '5px 13px', borderRadius: 7,
              border: `1px solid ${active === tab.key ? tab.color + '50' : 'var(--border)'}`,
              background: active === tab.key ? tab.color + '15' : 'transparent',
              color: active === tab.key ? tab.color : 'var(--text-muted)',
              fontFamily: 'var(--font-mono)', fontSize: 11.5,
              cursor: 'pointer', transition: 'all 0.18s',
              fontWeight: active === tab.key ? 600 : 400,
            }}
          >{tab.label}</button>
        ))}

        {/* Download button */}
        {src && (
          <button
            onClick={() => downloadB64Image(src, `${activeTab.key}_lane.png`)}
            style={{
              marginLeft: 'auto', padding: '5px 13px', borderRadius: 7,
              border: '1px solid var(--border)', background: 'transparent',
              color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
              fontSize: 11.5, cursor: 'pointer', transition: 'all 0.18s',
              display: 'flex', alignItems: 'center', gap: 5,
            }}
          >
            <DownloadIcon /> Save
          </button>
        )}
      </div>

      {/* Image */}
      {src && (
        <div key={activeTab.key} className="slide-up" style={{
          borderRadius: 'var(--radius-lg)',
          overflow: 'hidden',
          border: '1px solid var(--border)',
          background: 'var(--bg-elevated)',
          position: 'relative',
        }}>
          <img
            src={b64ToDataUri(src)}
            alt={activeTab.label}
            style={{ width: '100%', display: 'block', objectFit: 'contain' }}
          />
          {/* Label badge */}
          <div style={{
            position: 'absolute', top: 10, left: 10,
            padding: '3px 10px', borderRadius: 100,
            background: 'rgba(10,10,8,0.75)',
            border: `1px solid ${activeTab.color}40`,
            fontFamily: 'var(--font-mono)', fontSize: 10.5,
            color: activeTab.color, backdropFilter: 'blur(6px)',
          }}>
            {activeTab.label}
          </div>
        </div>
      )}
    </div>
  );
}

function DownloadIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}