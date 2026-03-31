import { useState, useRef, useEffect } from 'react';

interface Option {
  value: number;
  label: string;
}

interface Props {
  options: Option[];
  value: number;
  onChange: (val: number) => void;
  placeholder?: string;
}

export default function CustomSelect({ options, value, onChange, placeholder = '— Selecciona —' }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const selected = options.find(o => o.value === value);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <div className="custom-select" ref={ref}>
      <div
        className={`custom-select-trigger${open ? ' open' : ''}`}
        onClick={() => setOpen(!open)}
      >
        <span style={{ color: selected ? '#f0f0f5' : 'rgba(240,240,245,0.45)' }}>
          {selected ? selected.label : placeholder}
        </span>
        <svg width="12" height="8" viewBox="0 0 12 8" style={{ flexShrink: 0, transition: 'transform 0.2s', transform: open ? 'rotate(180deg)' : 'none' }}>
          <path d="M1 1l5 5 5-5" stroke="#999" fill="none" strokeWidth="2" />
        </svg>
      </div>
      {open && (
        <div className="custom-select-dropdown">
          {options.map(o => (
            <div
              key={o.value}
              onClick={() => { onChange(o.value); setOpen(false); }}
              style={{
                padding: '0.65rem 1rem',
                fontSize: '14px',
                cursor: 'pointer',
                color: o.value === value ? '#ff6b4a' : '#f0f0f5',
                background: o.value === value ? 'rgba(232,70,42,0.15)' : 'transparent',
                fontWeight: o.value === value ? 600 : 400,
                transition: 'background 0.1s',
              }}
              onMouseEnter={e => { if (o.value !== value) (e.currentTarget as HTMLDivElement).style.background = 'rgba(255,255,255,0.06)'; }}
              onMouseLeave={e => { if (o.value !== value) (e.currentTarget as HTMLDivElement).style.background = 'transparent'; }}
            >
              {o.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
