import type { SubtitleStyleState } from '../types';

export function SubtitleStyleControls({
  value,
  onChange,
}: {
  value: SubtitleStyleState;
  onChange: (style: SubtitleStyleState) => void;
}) {
  const update = <Key extends keyof SubtitleStyleState>(
    key: Key,
    nextValue: SubtitleStyleState[Key],
  ) => onChange({ ...value, [key]: nextValue });

  return (
    <div className="subtitle-style-grid">
      <label className="config-field">
        <span>Fuente de subtítulos</span>
        <select className="text-input" value={value.font_family ?? ''} onChange={(event) => {
          update('font_family', event.target.value || null);
        }}>
          <option value="">Usar fuente de plantilla</option>
          <option value="Arial">Arial</option>
          <option value="Arial Black">Arial Black</option>
          <option value="Georgia">Georgia</option>
          <option value="Tahoma">Tahoma</option>
          <option value="Verdana">Verdana</option>
        </select>
      </label>
      <label className="config-field">
        <span>Tamaño de fuente</span>
        <select className="text-input" value={value.font_size ?? ''} onChange={(event) => {
          update('font_size', event.target.value ? Number(event.target.value) : null);
        }}>
          <option value="">Usar tamaño de plantilla</option>
          <option value="32">Pequeño</option>
          <option value="48">Mediano</option>
          <option value="64">Grande</option>
          <option value="80">Muy grande</option>
        </select>
      </label>
      <div className="subtitle-color-control">
        <label htmlFor="subtitle-font-color">Color de texto</label>
        <input
          id="subtitle-font-color"
          type="color"
          value={value.font_color ?? '#ffffff'}
          onChange={(event) => update('font_color', event.target.value)}
        />
        <button type="button" onClick={() => update('font_color', null)}>Usar plantilla</button>
      </div>
      <div className="subtitle-color-control">
        <label htmlFor="subtitle-highlight-color">Color de resaltado</label>
        <input
          id="subtitle-highlight-color"
          type="color"
          value={value.highlight_color ?? '#ffff00'}
          onChange={(event) => update('highlight_color', event.target.value)}
        />
        <button type="button" onClick={() => update('highlight_color', null)}>Usar plantilla</button>
      </div>
      <label className="config-field">
        <span>Posición</span>
        <select className="text-input" value={value.position ?? ''} onChange={(event) => {
          const position = event.target.value;
          update(
            'position',
            position === 'top' || position === 'center' || position === 'bottom'
              ? position
              : null,
          );
        }}>
          <option value="">Usar posición de plantilla</option>
          <option value="top">Arriba</option>
          <option value="center">Centro</option>
          <option value="bottom">Abajo</option>
        </select>
      </label>
    </div>
  );
}
