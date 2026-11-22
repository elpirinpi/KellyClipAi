// Formulario de la plantilla de marca (Fase 3, ítem 5): logo, colores
// y fuente de marca. Un kit por usuario; se aplica a los clips del
// proyecto con la opción "Aplicar mi marca" y en el Studio con el
// botón "Aplicar mi marca" de la pestaña Encuadre.
import { useEffect, useRef, useState } from 'react';
import { Check, ImagePlus, LoaderCircle, Trash2 } from 'lucide-react';
import { API_BASE } from '../context/constants';
import {
  deleteBrandKit,
  fetchBrandKit,
  saveBrandKit,
  uploadBrandLogo,
  type BrandKit,
} from '../studio/api';

const BRAND_FONTS = ['Arial', 'Arial Black', 'Georgia', 'Tahoma', 'Verdana'] as const;

export function BrandKitForm({ notify }: { notify: (message: string, kind?: 'info' | 'error' | 'success') => void }) {
  const [kit, setKit] = useState<BrandKit | null>(null);
  const [loading, setLoading] = useState(true);
  const [name, setName] = useState('');
  const [primaryColor, setPrimaryColor] = useState('#ffffff');
  const [accentColor, setAccentColor] = useState('#facc15');
  const [fontFamily, setFontFamily] = useState<string>('');
  const [logoFile, setLogoFile] = useState<File | null>(null);
  const [logoPreview, setLogoPreview] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const logoInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchBrandKit()
      .then((loaded) => {
        if (cancelled) return;
        setKit(loaded);
        if (loaded) {
          setName(loaded.name);
          setPrimaryColor(loaded.primary_color);
          setAccentColor(loaded.accent_color);
          setFontFamily(loaded.font_family ?? '');
        }
      })
      .catch(() => {
        if (!cancelled) notify('No se pudo cargar la plantilla de marca.', 'error');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [notify]);

  useEffect(() => {
    if (!logoFile) {
      setLogoPreview(null);
      return;
    }
    const url = URL.createObjectURL(logoFile);
    setLogoPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [logoFile]);

  const handleSave = async () => {
    setSaving(true);
    try {
      const saved = await saveBrandKit({
        name: name.trim(),
        primary_color: primaryColor,
        accent_color: accentColor,
        font_family: fontFamily || null,
      });
      let current = saved;
      if (logoFile) {
        current = await uploadBrandLogo(logoFile);
        setLogoFile(null);
      }
      setKit(current);
      notify('Plantilla de marca guardada.', 'success');
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo guardar la plantilla de marca.', 'error');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!window.confirm('¿Borrar la plantilla de marca (incluido el logo)?')) return;
    setDeleting(true);
    try {
      await deleteBrandKit();
      setKit(null);
      setName('');
      setPrimaryColor('#ffffff');
      setAccentColor('#facc15');
      setFontFamily('');
      setLogoFile(null);
      notify('Plantilla de marca borrada.', 'success');
    } catch (error) {
      notify(error instanceof Error ? error.message : 'No se pudo borrar la plantilla de marca.', 'error');
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <p className="empty-templates">
        <LoaderCircle size={15} className="spin" /> Cargando plantilla de marca…
      </p>
    );
  }

  const logoSrc = logoPreview ?? (kit?.logo_url ? `${API_BASE}${kit.logo_url}` : null);

  return (
    <div className="brand-form">
      <p className="brand-form-intro">
        Tu identidad visual en todos los clips: el logo se superpone en la
        esquina y los colores tiñen los subtítulos. Activa “Aplicar mi marca”
        en las opciones de edición para usarla en este proyecto.
      </p>
      <div className="brand-form-grid">
        <label className="config-field">
          <span>Nombre</span>
          <input
            className="text-input"
            type="text"
            maxLength={60}
            placeholder="Mi canal"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <label className="config-field">
          <span>Fuente de marca</span>
          <select
            className="text-input"
            value={fontFamily}
            onChange={(event) => setFontFamily(event.target.value)}
          >
            <option value="">La de la plantilla</option>
            {BRAND_FONTS.map((font) => (
              <option key={font} value={font}>{font}</option>
            ))}
          </select>
        </label>
        <label className="config-field">
          <span>Color principal (subtítulos)</span>
          <span className="brand-color-row">
            <input
              type="color"
              value={primaryColor}
              onChange={(event) => setPrimaryColor(event.target.value)}
              aria-label="Color principal"
            />
            <input
              className="text-input"
              type="text"
              maxLength={7}
              value={primaryColor}
              onChange={(event) => setPrimaryColor(event.target.value)}
            />
          </span>
        </label>
        <label className="config-field">
          <span>Color de acento (resaltado)</span>
          <span className="brand-color-row">
            <input
              type="color"
              value={accentColor}
              onChange={(event) => setAccentColor(event.target.value)}
              aria-label="Color de acento"
            />
            <input
              className="text-input"
              type="text"
              maxLength={7}
              value={accentColor}
              onChange={(event) => setAccentColor(event.target.value)}
            />
          </span>
        </label>
      </div>
      <div className="brand-logo-row">
        <input
          ref={logoInputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="visually-hidden"
          onChange={(event) => {
            const file = event.target.files?.[0] ?? null;
            setLogoFile(file);
            event.target.value = '';
          }}
        />
        <button
          type="button"
          className="subtle-button"
          onClick={() => logoInputRef.current?.click()}
        >
          <ImagePlus size={15} /> {kit?.logo_filename || logoFile ? 'Cambiar logo' : 'Subir logo'}
        </button>
        {logoSrc && (
          <span className="brand-logo-preview">
            <img src={logoSrc} alt="Logo de marca" />
            <span>{logoFile ? logoFile.name : kit?.logo_filename}</span>
          </span>
        )}
        <span className="brand-logo-hint">PNG, JPG o WebP, máx. 2 MB.</span>
      </div>
      <div className="brand-form-actions">
        <button type="button" className="create-button" onClick={() => void handleSave()} disabled={saving}>
          {saving ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />}
          {saving ? 'Guardando…' : 'Guardar marca'}
        </button>
        {kit && (
          <button type="button" className="delete-project-button" onClick={() => void handleDelete()} disabled={deleting}>
            <Trash2 size={15} /> {deleting ? 'Borrando…' : 'Borrar marca'}
          </button>
        )}
      </div>
    </div>
  );
}
