// UploadPanel.jsx – Drag & drop image uploader

import { useState, useRef, useCallback } from 'react';

const ACCEPTED      = ['.png', '.jpg', '.jpeg', '.bmp'];
const ACCEPTED_MIME = ['image/png', 'image/jpeg', 'image/bmp'];

export default function UploadPanel({ onImageSelect, onMaskSelect, disabled = false }) {
  const [isDragging,    setIsDragging]    = useState(false);
  const [imageFile,     setImageFile]     = useState(null);
  const [maskFile,      setMaskFile]      = useState(null);
  const [imagePreview,  setImagePreview]  = useState(null);
  const [withMask,      setWithMask]      = useState(false);
  const [error,         setError]         = useState('');

  const imageRef = useRef();
  const maskRef  = useRef();

  const validate = (file) => {
    if (!file) return 'No file selected.';
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ACCEPTED.includes(ext) && !ACCEPTED_MIME.includes(file.type))
      return `Unsupported format. Accepted: ${ACCEPTED.join(', ')}`;
    if (file.size > 16 * 1024 * 1024) return 'File too large. Max 16 MB.';
    return null;
  };

  const handleImage = useCallback((file) => {
    setError('');
    const err = validate(file);
    if (err) { setError(err); return; }
    setImageFile(file);
    onImageSelect(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.onerror = () => setImagePreview(null);
    reader.readAsDataURL(file);
  }, [onImageSelect]);

  const handleMask = useCallback((file) => {
    setError('');
    const err = validate(file);
    if (err) { setError(err); return; }
    setMaskFile(file);
    onMaskSelect && onMaskSelect(file);
  }, [onMaskSelect]);

  const onDragOver  = (e) => { e.preventDefault(); if (!disabled) setIsDragging(true); };
  const onDragLeave = ()  => setIsDragging(false);
  const onDrop      = (e) => {
    e.preventDefault(); setIsDragging(false);
    if (disabled) return;
    handleImage(e.dataTransfer.files[0]);
  };

  const clearAll = () => {
    setImageFile(null); setMaskFile(null);
    setImagePreview(null); setError(''); setWithMask(false);
    onImageSelect(null);
    onMaskSelect && onMaskSelect(null);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

      {/* Drop zone */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !disabled && !imageFile && imageRef.current.click()}
        style={{
          position: 'relative',
          border: `2px dashed ${
            isDragging ? 'var(--accent)' :
            imageFile  ? 'var(--green)'  :
            error      ? 'var(--red)'    : 'var(--border)'}`,
          borderRadius: 'var(--radius-lg)',
          background: isDragging
            ? 'rgba(245,158,11,0.06)'
            : imageFile
            ? 'rgba(34,197,94,0.04)'
            : 'var(--bg-card)',
          padding: imageFile ? '16px' : '44px 24px',
          textAlign: 'center',
          cursor: disabled ? 'not-allowed' : imageFile ? 'default' : 'pointer',
          transition: 'all 0.25s ease',
          opacity: disabled ? 0.5 : 1,
          overflow: 'hidden',
        }}
      >
        {/* Scanning line when dragging */}
        {isDragging && (
          <div style={{
            position: 'absolute', left: 0, right: 0, height: 2,
            background: 'linear-gradient(90deg, transparent, var(--accent), transparent)',
            animation: 'scan-h 1s ease-in-out infinite',
            pointerEvents: 'none',
          }} />
        )}

        {imageFile ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{
              width: 72, height: 52, flexShrink: 0,
              borderRadius: 8, overflow: 'hidden',
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              {imagePreview
                ? <img src={imagePreview} alt="preview"
                       style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                : <ImgIcon />}
            </div>
            <div style={{ flex: 1, textAlign: 'left', minWidth: 0 }}>
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 13,
                color: 'var(--green)', fontWeight: 500,
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>{imageFile.name}</div>
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 11,
                color: 'var(--text-muted)', marginTop: 3,
              }}>{(imageFile.size / 1024).toFixed(1)} KB</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%',
                               background: 'var(--green)', display: 'inline-block' }} />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5,
                               color: 'var(--green)' }}>Ready for detection</span>
              </div>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); imageRef.current.click(); }}
              disabled={disabled}
              style={{
                padding: '6px 14px', borderRadius: 8,
                border: '1px solid var(--border)', background: 'transparent',
                color: 'var(--text-secondary)',
                fontFamily: 'var(--font-body)', fontSize: 12, cursor: 'pointer',
              }}
            >Replace</button>
          </div>
        ) : (
          <>
            <div style={{
              width: 52, height: 52, margin: '0 auto 16px',
              borderRadius: 12,
              background: isDragging ? 'var(--accent-glow)' : 'var(--bg-elevated)',
              border: `1px solid ${isDragging ? 'var(--accent-border)' : 'var(--border)'}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.2s',
            }}>
              <UploadIcon color={isDragging ? 'var(--accent)' : 'var(--text-muted)'} />
            </div>
            <div style={{
              fontFamily: 'var(--font-heading)', fontSize: 15, fontWeight: 600,
              color: isDragging ? 'var(--accent)' : 'var(--text-primary)',
              marginBottom: 6, transition: 'color 0.2s',
            }}>
              {isDragging ? 'Drop driving image here' : 'Upload Driving Image'}
            </div>
            <div style={{
              fontFamily: 'var(--font-body)', fontSize: 13,
              color: 'var(--text-muted)', marginBottom: 16,
            }}>
              Drag & drop or{' '}
              <span style={{ color: 'var(--accent)', fontWeight: 500 }}>click to browse</span>
            </div>
            <div style={{ display: 'flex', gap: 6, justifyContent: 'center', flexWrap: 'wrap' }}>
              {ACCEPTED.map(ext => (
                <span key={ext} style={{
                  padding: '2px 9px', borderRadius: 100,
                  border: '1px solid var(--border)',
                  fontFamily: 'var(--font-mono)', fontSize: 10.5,
                  color: 'var(--text-muted)', textTransform: 'uppercase',
                }}>{ext.replace('.', '')}</span>
              ))}
            </div>
          </>
        )}
        <input ref={imageRef} type="file" accept={ACCEPTED.join(',')}
               style={{ display: 'none' }}
               onChange={(e) => handleImage(e.target.files[0])} />
      </div>

      {/* Error */}
      {error && (
        <div style={{
          padding: '10px 14px', borderRadius: 'var(--radius-md)',
          background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)',
          fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--red)',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>⚠ {error}</div>
      )}

      {/* Ground-truth mask toggle */}
      {imageFile && onMaskSelect && (
        <div style={{
          padding: '14px 16px', borderRadius: 'var(--radius-md)',
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
        }}>
          <label style={{
            display: 'flex', alignItems: 'center',
            justifyContent: 'space-between', cursor: 'pointer',
          }}>
            <div>
              <div style={{ fontFamily: 'var(--font-body)', fontSize: 13.5,
                            fontWeight: 500, color: 'var(--text-primary)' }}>
                Upload ground-truth mask
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11,
                            color: 'var(--text-muted)', marginTop: 2 }}>
                Enables Dice / IoU metric comparison
              </div>
            </div>
            <div
              onClick={() => { setWithMask(!withMask); if (withMask) { setMaskFile(null); onMaskSelect(null); } }}
              style={{
                width: 42, height: 24, borderRadius: 100,
                background: withMask ? 'var(--accent)' : 'var(--bg-surface)',
                border: `1px solid ${withMask ? 'var(--accent)' : 'var(--border)'}`,
                position: 'relative', flexShrink: 0,
                cursor: 'pointer', transition: 'all 0.2s',
              }}
            >
              <div style={{
                position: 'absolute', top: 3,
                left: withMask ? 20 : 3,
                width: 16, height: 16, borderRadius: '50%',
                background: 'white', transition: 'left 0.2s',
                boxShadow: '0 1px 4px rgba(0,0,0,0.3)',
              }} />
            </div>
          </label>

          {withMask && (
            <div style={{ marginTop: 12 }}>
              <button
                onClick={() => maskRef.current.click()}
                disabled={disabled}
                style={{
                  width: '100%', padding: '10px', borderRadius: 10,
                  border: `1px dashed ${maskFile ? 'var(--green)' : 'var(--border)'}`,
                  background: maskFile ? 'rgba(34,197,94,0.06)' : 'transparent',
                  color: maskFile ? 'var(--green)' : 'var(--text-secondary)',
                  fontFamily: 'var(--font-mono)', fontSize: 12,
                  cursor: 'pointer', transition: 'all 0.2s',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                }}
              >
                {maskFile ? `✓ ${maskFile.name}` : '↑ Click to select mask file'}
              </button>
              <input ref={maskRef} type="file" accept={ACCEPTED.join(',')}
                     style={{ display: 'none' }}
                     onChange={(e) => handleMask(e.target.files[0])} />
            </div>
          )}
        </div>
      )}

      {/* Clear */}
      {imageFile && (
        <button onClick={clearAll} disabled={disabled} style={{
          alignSelf: 'flex-start', padding: '6px 16px', borderRadius: 8,
          border: '1px solid var(--border)', background: 'transparent',
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-body)', fontSize: 12, cursor: 'pointer',
        }}>✕ Clear</button>
      )}
    </div>
  );
}

function UploadIcon({ color = 'currentColor', size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke={color} strokeWidth="1.8" strokeLinecap="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}
function ImgIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
         stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round">
      <rect x="3" y="3" width="18" height="18" rx="2"/>
      <circle cx="8.5" cy="8.5" r="1.5"/>
      <polyline points="21 15 16 10 5 21"/>
    </svg>
  );
}