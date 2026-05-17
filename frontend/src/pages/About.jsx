// About.jsx – Model info & dataset documentation page

export default function About({ modelInfo, serverStatus }) {
  const isOnline = serverStatus === 'online';
  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '40px 2rem 72px' }}>

      {/* Header */}
      <div className="fade-in" style={{ marginBottom: 48 }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          padding: '4px 12px', borderRadius: 100,
          background: 'rgba(96,165,250,0.10)', border: '1px solid rgba(96,165,250,0.22)',
          marginBottom: 16,
        }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11,
                         color: 'var(--blue)', letterSpacing: '0.06em',
                         textTransform: 'uppercase' }}>Model Documentation</span>
        </div>
        <h1 style={{
          fontFamily: 'var(--font-heading)',
          fontSize: 'clamp(26px, 3.5vw, 40px)',
          fontWeight: 800, letterSpacing: '-0.03em',
          lineHeight: 1.1, marginBottom: 12,
        }}>
          U-Net Architecture &{' '}
          <span style={{
            background: 'linear-gradient(135deg, #f59e0b, #fcd34d)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          }}>KITTI Dataset</span>
        </h1>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: 15,
                    color: 'var(--text-secondary)', maxWidth: 540, lineHeight: 1.7 }}>
          Technical reference for the lane detection model, training dataset, and evaluation metrics.
        </p>
      </div>

      {/* Live stats */}
      <div className="card" style={{ padding: '20px 24px', marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 18 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%',
                         background: isOnline ? 'var(--green)' : 'var(--red)',
                         display: 'inline-block' }} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12,
                         color: 'var(--text-muted)' }}>
            Live model stats — {isOnline ? 'server online' : 'server offline'}
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {[
            { label: 'Architecture',    value: modelInfo?.architecture        || 'U-Net (ResNet34)' },
            { label: 'Parameters',      value: modelInfo?.trainable_params
                                               ? `${(modelInfo.trainable_params / 1e6).toFixed(1)}M`
                                               : '~24M' },
            { label: 'Input Size',      value: '224 × 224 × 3' },
            { label: 'Mask Threshold',  value: modelInfo?.mask_threshold?.toString() || '0.5' },
          ].map(({ label, value }) => (
            <div key={label} style={{
              padding: '14px 12px', borderRadius: 10,
              background: 'var(--bg-elevated)', border: '1px solid var(--border)',
            }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10,
                            color: 'var(--text-muted)', marginBottom: 6,
                            textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
              <div style={{ fontFamily: 'var(--font-heading)', fontSize: 15,
                            fontWeight: 700, color: 'var(--accent)' }}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Two cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>

        {/* Architecture */}
        <div className="card" style={{ padding: 24 }}>
          <CardLabel text="Model Architecture" color="var(--accent)" />
          <p style={{ fontFamily: 'var(--font-body)', fontSize: 13.5,
                      color: 'var(--text-secondary)', lineHeight: 1.75, marginBottom: 16 }}>
            The U-Net uses a pretrained ResNet34 encoder from
            <code style={{ fontFamily: 'var(--font-mono)', fontSize: 12,
                           color: 'var(--accent)', padding: '0 4px' }}>segmentation-models-pytorch</code>.
            The decoder progressively upsamples the feature maps using transposed
            convolutions with skip connections from each encoder level.
          </p>
          <InfoRow label="Encoder"    value="ResNet34 (pretrained on ImageNet)" />
          <InfoRow label="Decoder"    value="5× upsample blocks with skip connections" />
          <InfoRow label="Output"     value="Binary mask — sigmoid activation" />
          <InfoRow label="Loss"       value="BCE + Dice combined loss" />
          <InfoRow label="Optimizer"  value="Adam, lr=1e-4" />
        </div>

        {/* Dataset */}
        <div className="card" style={{ padding: 24 }}>
          <CardLabel text="KITTI Dataset" color="var(--blue)" />
          <p style={{ fontFamily: 'var(--font-body)', fontSize: 13.5,
                      color: 'var(--text-secondary)', lineHeight: 1.75, marginBottom: 16 }}>
            The KITTI Road/Lane Detection dataset contains real driving footage
            with manually annotated road and lane masks. Images are pre-resized
            to 224×224 pixels for this project.
          </p>
          <InfoRow label="Source"      value="Kaggle — KITTI Road/Lane 224×224" />
          <InfoRow label="Images"      value="Training + Testing splits" />
          <InfoRow label="Mask format" value="Pink/magenta pixels = lane region" />
          <InfoRow label="Resolution"  value="224 × 224 pixels" />
          <InfoRow label="Task"        value="Binary lane segmentation (IoU)" />
        </div>
      </div>

      {/* Metrics explanation */}
      <div className="card" style={{ padding: 24, marginTop: 20 }}>
        <CardLabel text="Evaluation Metrics" color="var(--green)" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 16 }}>
          {[
            { name: 'Dice Score', formula: '2·TP / (2·TP + FP + FN)', desc: 'Harmonic mean of precision and recall. Ranges 0–1; higher is better.' },
            { name: 'IoU (Jaccard)', formula: 'TP / (TP + FP + FN)', desc: 'Intersection over Union. Stricter than Dice; standard metric for segmentation.' },
            { name: 'Pixel Accuracy', formula: '(TP + TN) / total', desc: 'Fraction of correctly classified pixels. Can be misleading with class imbalance.' },
          ].map(({ name, formula, desc }) => (
            <div key={name} style={{
              padding: '16px 14px', borderRadius: 10,
              background: 'var(--bg-elevated)', border: '1px solid var(--border)',
            }}>
              <div style={{ fontFamily: 'var(--font-heading)', fontSize: 14,
                            fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>{name}</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11,
                            color: 'var(--accent)', marginBottom: 8 }}>{formula}</div>
              <div style={{ fontFamily: 'var(--font-body)', fontSize: 12.5,
                            color: 'var(--text-secondary)', lineHeight: 1.6 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function CardLabel({ text, color }) {
  return (
    <div style={{ fontFamily: 'var(--font-heading)', fontSize: 16, fontWeight: 700,
                  color, marginBottom: 14 }}>{text}</div>
  );
}
function InfoRow({ label, value }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'baseline', gap: 10,
      padding: '7px 0', borderBottom: '1px solid var(--border-subtle)',
    }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11,
                     color: 'var(--text-muted)', flexShrink: 0, minWidth: 100 }}>{label}</span>
      <span style={{ fontFamily: 'var(--font-body)', fontSize: 13,
                     color: 'var(--text-secondary)' }}>{value}</span>
    </div>
  );
}