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
        <span className={selected ? '' : 'placeholder'}>{selected ? selected.label : placeholder}</span>
        <svg width="12" height="8" viewBox="0 0 12 8" className={`chevron${open ? ' flipped' : ''}`}>
          <path d="M1 1l5 5 5-5" stroke="#666" fill="none" strokeWidth="2" />
        </svg>
      </div>
      {open && (
        <div className="custom-select-dropdown">
          {options.map(o => (
            <div
              key={o.value}
              className={`custom-select-option${o.value === value ? ' selected' : ''}`}
              onClick={() => { onChange(o.value); setOpen(false); }}
            >
              {o.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
