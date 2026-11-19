import {
  AlertCircle, ArrowRight, AudioLines, CalendarDays, Check, ChevronDown, ChevronLeft, ChevronRight, CloudUpload, FileVideo2, FolderOpen, Link2, LoaderCircle, LockKeyhole, Plus, Video, WandSparkles, X,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { formatBytes } from '../context/constants';
import { ACCEPT, MAX_UPLOAD_LABEL } from '../hooks/useImport';
import { SubtitleStyleControls } from './SubtitleStyleControls';
import { TemplatePreview } from './TemplatePreview';
import { formatDuration } from '../utils';

// Página principal: subida/importación y configuración de clips
// (extraída de App.tsx, ítem 23 de la Fase 2).
export function HomePage() {
  const {
    navigate,
    linkProject,
    isSubmitting,
    project,
    projectError,
    projectRouteId,
    projectRouteMode,
    notify,
    submitProject,
    config: {
      ratio, setRatio, duration, setDuration, framing, setFraming,
      activeTemplate, setActiveTemplate, templateTab, setTemplateTab,
      savedTemplates, templateName, setTemplateName, savingTemplate,
      options, toggleOption, moment, setMoment,
      schedule, setSchedule, scheduleDate, setScheduleDate,
      schedulePlatform, setSchedulePlatform, scheduleVisibility, setScheduleVisibility,
      whisperLanguage, setWhisperLanguage, subtitleStyle, setSubtitleStyle,
      visibleTemplates, saveCurrentTemplate, onTemplateTabKeyDown,
    },
    importer: {
      fileInfo, thumbnail, progress, uploadError, uploadAttempt,
      dragging, setDragging, inputRef, chooseFile, onDrop, cancelImport,
    },
    linkImport: {
      link, setLink, linkMessage, setLinkMessage, downloadProgress, handleLink, cancelLinkDownload,
    },
    recorder: { setRecordOpen, setRecordError },
    zoom: { openZoomImport },
  } = useApp();

  return (
    <main className="main-content" id="main-content">
      <h1 className="visually-hidden">Crear clips con IA</h1>
      {!projectRouteId && <section className="upload-stack" aria-label="Subir video">
        {progress !== null && (
          <div className="upload-progress-card" aria-live="polite">
            <div className="upload-thumb">
              {thumbnail ? <img src={thumbnail} alt="Miniatura del archivo" /> : fileInfo?.kind === 'audio' ? <AudioLines size={22} /> : <FileVideo2 size={22} />}
            </div>
            <div className="upload-details">
              <div className="upload-file-head">
                <span className="file-name" title={fileInfo?.name}>{fileInfo?.name ?? 'Preparando archivo…'}</span>
                {fileInfo?.width && fileInfo.height ? <span className="meta-pill">{Math.max(fileInfo.width, fileInfo.height) >= 1900 ? '1080p' : `${Math.max(fileInfo.width, fileInfo.height)}px`}</span> : null}
                {fileInfo && <span className="meta-pill">{formatDuration(fileInfo.duration)}</span>}
                <button className="icon-button cancel-upload" type="button" onClick={cancelImport} aria-label="Cancelar importación" disabled={progress === 100}><X size={17} /></button>
              </div>
              <div className="progress-line"><span style={{ width: `${progress}%` }} /></div>
              <div className="progress-copy"><span>{progress === 100 ? 'Listo en el almacenamiento local' : uploadAttempt && uploadAttempt.attempt > 1 ? `Reintentando subida · intento ${uploadAttempt.attempt} de ${uploadAttempt.maxAttempts} · ${progress}%` : `Subiendo al almacenamiento local ${progress}%`}</span><span>{progress}%</span></div>
            </div>
          </div>
        )}

        <form className="link-form" onSubmit={handleLink}>
          <Link2 size={18} className="link-leading" aria-hidden="true" />
          <input
            aria-label="Enlace de video"
            type="url"
            value={link}
            onChange={(event) => { setLink(event.target.value); setLinkMessage(''); }}
            placeholder="Pega el enlace de un video"
          />
          <button type="submit" className="continue-button" disabled={!link.trim() || isSubmitting}>{isSubmitting ? 'Descargando…' : 'Continuar'} <ArrowRight size={16} /></button>
        </form>
        {linkMessage && <p className="form-message" role="status">{linkMessage}</p>}
        {downloadProgress?.source_type === 'url' && downloadProgress.status === 'downloading' && (
          <div className="download-progress" aria-live="polite">
            {downloadProgress.total_bytes && downloadProgress.total_bytes > 0 && (
              <div className="progress-line">
                <span style={{ width: `${Math.min(100, Math.round((downloadProgress.bytes_received / downloadProgress.total_bytes) * 100))}%` }} />
              </div>
            )}
            <div className="progress-copy">
              <span>
                {downloadProgress.total_bytes && downloadProgress.total_bytes > 0
                  ? `${formatBytes(downloadProgress.bytes_received)} de ${formatBytes(downloadProgress.total_bytes)}`
                  : `${formatBytes(downloadProgress.bytes_received)} descargados`}
                {downloadProgress.speed ? ` · ${formatBytes(downloadProgress.speed)}/s` : ''}
                {downloadProgress.eta !== null ? ` · quedan ${formatDuration(downloadProgress.eta)}` : ''}
              </span>
              <span>
                {downloadProgress.total_bytes && downloadProgress.total_bytes > 0
                  ? `${Math.min(100, Math.round((downloadProgress.bytes_received / downloadProgress.total_bytes) * 100))}%`
                  : ''}
              </span>
            </div>
            <button type="button" className="cancel-download-button" onClick={() => void cancelLinkDownload()}>
              <X size={14} /> Cancelar descarga
            </button>
          </div>
        )}

        <div
          className={`drop-zone ${dragging ? 'is-dragging' : ''}`}
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inputRef.current?.click(); } }}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input ref={inputRef} className="visually-hidden" type="file" accept={ACCEPT} aria-label="Seleccionar video o audio" onChange={chooseFile} />
          <span className="upload-illustration"><CloudUpload size={30} strokeWidth={1.7} /></span>
          <p className="drop-title">Haz clic para explorar o arrastra y suelta</p>
          <p className="drop-subtitle">Sube un video o audio desde tu dispositivo</p>
          <div className="accepted-types">
            <span>MP4</span><span>MOV</span><span>3GP</span><span>AVI</span><span>MP3</span><span>WAV</span><span>M4A</span>
          </div>
          <p className="size-note">Tus archivos permanecen en este dispositivo · Tamaño máximo: {MAX_UPLOAD_LABEL}</p>
        </div>

        {uploadError && <p className="error-message" role="alert"><AlertCircle size={16} />{uploadError}</p>}

        <div className="secondary-actions">
          <button type="button" className="outline-action" onClick={() => void openZoomImport()}>
            <span className="zoom-mark">Z</span> Importar desde Zoom
          </button>
          <button type="button" className="outline-action" onClick={() => { setRecordError(''); setRecordOpen(true); }}>
            <Video size={18} className="record-icon" /> Grabar video
          </button>
        </div>
        <p className="authorization-note">Importa solo videos propios o para los que tengas autorización.</p>
        {/* Fase 2 (ítem 15): acceso al historial desde la página principal. */}
        <button type="button" className="projects-home-link" onClick={() => navigate('/projects')}>
          <FolderOpen size={15} /> Ver mis proyectos
        </button>
      </section>}

      {((!projectRouteId && fileInfo && progress === 100) || (projectRouteMode === 'configure' && project)) && (
        <section className="config-area" aria-label="Configuración de clips">
          {projectRouteId && project && <div className="project-source-card"><FileVideo2 size={20} /><span><strong>{project.title}</strong><small>Video importado y guardado en este dispositivo</small></span><span className="meta-pill">{project.metadata ? formatDuration(project.metadata.duration) : 'Listo'}</span></div>}
          <div className="settings-row">
            <label className="config-field">
              <span>Ratio</span>
              <span className="select-wrap"><select value={ratio} onChange={(event) => setRatio(event.target.value)}><option>9:16</option><option>16:9</option><option>1:1</option><option>4:5</option></select><ChevronDown size={16} /></span>
            </label>
            <label className="config-field">
              <span>Duración del clip</span>
              <span className="select-wrap"><select value={duration} onChange={(event) => setDuration(event.target.value)}><option>Cualquiera</option><option>&lt;30s</option><option>30–60s</option><option>60–90s</option><option>90s–3min</option></select><ChevronDown size={16} /></span>
            </label>
          </div>

          <section className="config-card templates-card">
            <div className="card-heading">
              <div><h2>Plantillas de subtítulos</h2><p>Elige el estilo de tus clips</p></div>
              <div className="template-save-controls">
                <input
                  aria-label="Nombre de la plantilla"
                  maxLength={60}
                  placeholder={`Mi estilo ${savedTemplates.length + 1}`}
                  value={templateName}
                  onChange={(event) => setTemplateName(event.target.value)}
                />
                <button className="subtle-button" type="button" onClick={() => void saveCurrentTemplate()} disabled={savingTemplate}>
                  {savingTemplate ? <LoaderCircle size={15} className="spin" /> : <Plus size={15} />}
                  {savingTemplate ? 'Guardando…' : 'Guardar estilo'}
                </button>
              </div>
            </div>
            <div className="tabs" role="tablist" aria-label="Categorías de plantillas" onKeyDown={onTemplateTabKeyDown}>
              <button type="button" role="tab" id="template-tab-featured" aria-controls="template-panel-featured" aria-selected={templateTab === 'featured'} tabIndex={templateTab === 'featured' ? 0 : -1} className={templateTab === 'featured' ? 'selected' : ''} onClick={() => setTemplateTab('featured')}>Destacadas</button>
              <button type="button" role="tab" id="template-tab-custom" aria-controls="template-panel-custom" aria-selected={templateTab === 'custom'} tabIndex={templateTab === 'custom' ? 0 : -1} className={templateTab === 'custom' ? 'selected' : ''} onClick={() => setTemplateTab('custom')}>Mis plantillas</button>
              <button type="button" role="tab" id="template-tab-brand" aria-controls="template-panel-brand" aria-selected={templateTab === 'brand'} tabIndex={templateTab === 'brand' ? 0 : -1} className={templateTab === 'brand' ? 'selected' : ''} onClick={() => setTemplateTab('brand')}>Plantilla de marca <span className="diamond">◆</span></button>
            </div>
            <div role="tabpanel" id={`template-panel-${templateTab}`} aria-labelledby={`template-tab-${templateTab}`}>
            {templateTab === 'brand' ? (
              <div className="locked-brand"><span className="lock-badge"><LockKeyhole size={20} /></span><div><strong>Plantilla de marca</strong><p>Personaliza clips con tu identidad visual. Esta opción premium todavía no está disponible.</p></div><span className="premium-chip">◆ Premium</span></div>
            ) : visibleTemplates.length ? (
              <div className="template-rail-wrap">
                <button className="rail-arrow rail-prev" type="button" aria-label="Ver plantillas anteriores" onClick={(event) => event.currentTarget.parentElement?.querySelector('.template-rail')?.scrollBy({ left: -260, behavior: 'smooth' })}><ChevronLeft size={18} /></button>
                <div className="template-rail">
                  {visibleTemplates.map((template) => (
                    <button key={template.name} className={`template-option ${activeTemplate === template.name ? 'active' : ''}`} type="button" onClick={() => setActiveTemplate(template.name)} aria-pressed={activeTemplate === template.name}>
                      {/* Fase 2 (ítem 18): vista previa real con la fuente, el tamaño, el color y la posición de la plantilla. Los ajustes manuales de estilo solo se aplican a la plantilla activa, que es la que se renderizará. */}
                      <TemplatePreview
                        name={template.base_template ?? template.name}
                        tone={template.tone}
                        sample={template.sample}
                        overrides={activeTemplate === template.name ? subtitleStyle : null}
                      />
                      <span className="template-name">{template.name}{activeTemplate === template.name && <span className="template-check"><Check size={11} /></span>}</span>
                    </button>
                  ))}
                </div>
                <button className="rail-arrow rail-next" type="button" aria-label="Ver más plantillas" onClick={(event) => event.currentTarget.parentElement?.querySelector('.template-rail')?.scrollBy({ left: 260, behavior: 'smooth' })}><ChevronRight size={18} /></button>
              </div>
            ) : <p className="empty-templates">Aún no guardaste estilos propios. Selecciona una plantilla y pulsa “Guardar estilo”.</p>}
            </div>
          </section>

          <section className="config-card options-card">
            <div className="card-heading"><div><h2>Opciones de edición</h2><p>Personaliza el resultado de tus clips</p></div></div>
            <div className="option-grid">
              <Option checked={options.emojis} onClick={() => toggleOption('emojis')} title="Añadir emojis" description="Haz que tus subtítulos destaquen" />
              <Option checked={options.keywords} onClick={() => toggleOption('keywords')} title="Resaltar palabras clave" description="Destaca las ideas importantes" />
              <Option checked={options.broll} onClick={() => { toggleOption('broll'); if (!options.broll) notify('B-rolls es una función premium y necesita una clave de Pexels o Pixabay.'); }} title="Añadir B-rolls" description="Inserta imágenes de apoyo" premium />
              <Option checked={options.silence} onClick={() => toggleOption('silence')} title="Eliminar silencios" description="Recorta las pausas largas" />
              <Option checked={options.censor} onClick={() => toggleOption('censor')} title="Auto-censurar" description="Silencia y oculta palabrotas" />
              <Option checked={options.face_tracking} onClick={() => toggleOption('face_tracking')} title="Seguir rostro en el encuadre" description="Mantiene a la persona principal visible al recortar" />
            </div>
            <label className="config-field">
              <span>Encuadre inteligente</span>
              <select className="text-input" value={framing} onChange={(event) => {
                const value = event.target.value;
                if (value === 'speaker' || value === 'split' || value === 'blur' || value === 'center') setFraming(value);
              }}>
                <option value="speaker">Seguir al orador</option>
                <option value="split">Pantalla dividida</option>
                <option value="blur">Fondo desenfocado</option>
                <option value="center">Centrado</option>
              </select>
              <small>El seguimiento del orador estima actividad labial durante la voz detectada, localmente.</small>
            </label>
            <label className="config-field">
              <span>Idioma de transcripción</span>
              <select className="text-input" value={whisperLanguage} onChange={(event) => {
                setWhisperLanguage(event.target.value);
              }}>
                <option value="auto">Detectar automáticamente</option>
                <option value="es">Español</option>
                <option value="en">Inglés</option>
                <option value="pt">Portugués</option>
                <option value="fr">Francés</option>
                <option value="de">Alemán</option>
                <option value="it">Italiano</option>
                <option value="ja">Japonés</option>
                <option value="zh">Chino</option>
                <option value="ko">Coreano</option>
                <option value="ru">Ruso</option>
                <option value="ar">Árabe</option>
                <option value="hi">Hindi</option>
                <option value="nl">Neerlandés</option>
                <option value="pl">Polaco</option>
                <option value="tr">Turco</option>
                <option value="uk">Ucraniano</option>
              </select>
              <small>Se guarda en el proyecto; automático conserva la detección de Whisper.</small>
            </label>
            <SubtitleStyleControls value={subtitleStyle} onChange={setSubtitleStyle} />
          </section>

          <section className="config-card moment-card">
            <div className="card-heading"><div><h2>Encontrar momento</h2><p>Orienta el análisis hacia un tema concreto</p></div><span className="optional-chip">Opcional</span></div>
            <input className="text-input" value={moment} onChange={(event) => setMoment(event.target.value)} placeholder="Por ejemplo: Cuando alguien habla de una idea importante." />
          </section>

          <section className="config-card schedule-card">
            <div className="schedule-heading">
              <div className="schedule-icon"><CalendarDays size={18} /></div>
              <div className="schedule-title"><h2>Programar clips</h2><p>Prepara una publicación para más tarde</p></div>
              <button className={`switch ${schedule ? 'on' : ''}`} type="button" role="switch" aria-checked={schedule} aria-label="Programar clips" onClick={() => setSchedule(!schedule)}><span /></button>
            </div>
            {schedule && (
              <div className="schedule-fields">
                <label className="config-field">
                  <span>Fecha y hora</span>
                  <input
                    type="datetime-local"
                    required
                    value={scheduleDate}
                    min={new Date(Date.now() + 60_000 - new Date().getTimezoneOffset() * 60_000).toISOString().slice(0, 16)}
                    onChange={(event) => setScheduleDate(event.target.value)}
                  />
                </label>
                <label className="config-field">
                  <span>Plataforma</span>
                  <span className="select-wrap"><select value={schedulePlatform} onChange={(event) => {
                    const next = event.target.value;
                    setSchedulePlatform(next);
                    if (next === 'instagram') setScheduleVisibility('public');
                    else if (scheduleVisibility === 'unlisted' && next === 'tiktok') setScheduleVisibility('private');
                  }}><option value="youtube">YouTube</option><option value="tiktok">TikTok</option><option value="instagram">Instagram</option></select><ChevronDown size={16} /></span>
                </label>
                <label className="config-field">
                  <span>Visibilidad</span>
                  <span className="select-wrap"><select value={scheduleVisibility} onChange={(event) => setScheduleVisibility(event.target.value)} disabled={schedulePlatform === 'instagram'}>
                    {schedulePlatform !== 'tiktok' && <option value="unlisted">No listada</option>}
                    {schedulePlatform !== 'instagram' && <option value="private">Privada</option>}
                    <option value="public">Pública</option>
                  </select><ChevronDown size={16} /></span>
                </label>
                <div className="connected-accounts"><span>Publicación real</span><p>Si no hay una cuenta conectada al llegar la fecha, el MP4 se exportará localmente sin publicarse. <button type="button" onClick={() => navigate('/settings')}>Configurar integraciones</button></p></div>
              </div>
            )}
          </section>

          {projectError && <p className="error-message" role="alert"><AlertCircle size={16} />{projectError}</p>}
          <button className="create-button" type="button" onClick={() => void submitProject()} disabled={isSubmitting || (!linkProject && progress !== 100)}>
            {isSubmitting ? <LoaderCircle size={18} className="spin" /> : <WandSparkles size={18} />}
            {isSubmitting ? 'Preparando proyecto…' : 'Obtener clips con IA'}
            {!isSubmitting && <ArrowRight size={17} />}
          </button>
          <p className="local-storage-note">El original permanece en este dispositivo; el proyecto se procesa en el backend local.</p>
        </section>
      )}
    </main>
  );
}

function Option({ checked, onClick, title, description, premium = false }: { checked: boolean; onClick: () => void; title: string; description: string; premium?: boolean }) {
  return (
    <button className={`option-item ${checked ? 'checked' : ''}`} type="button" role="checkbox" aria-checked={checked} onClick={onClick}>
      <span className={`fake-checkbox ${checked ? 'checked' : ''}`}>{checked && <Check size={12} strokeWidth={3} />}</span>
      <span className="option-copy"><span className="option-title">{title}{premium && <span className="diamond"> ◆</span>}</span><span className="option-description">{description}</span></span>
    </button>
  );
}
