// Pruebas de normalizeStudioEdit para los campos nuevos de la Fase 3
// (texto de gancho y logo de marca).
//
// Sin navegador y sin dependencias: se ejecuta con
//   node src/studio/types/editor.test.ts
// (desde app/; Node 22.18+ ejecuta TypeScript directamente) o con
//   npm run test:editor
// Cada comprobación que falla lanza y el proceso sale con código 1.
// El archivo no se importa desde la app, así que no entra en el
// bundle; `tsc --noEmit` (npm run build) sí lo comprueba.
import {
  DEFAULT_SETTINGS,
  normalizeStudioEdit,
} from './editor.ts';

let checks = 0;
function check(condition: unknown, message: string): void {
  checks += 1;
  if (!condition) throw new Error(`editor.test.ts: ${message}`);
}

// Una spec guardada con gancho y logo conserva ambos.
const saved = normalizeStudioEdit({
  version: 1,
  settings: {
    ...DEFAULT_SETTINGS,
    hookText: 'Lo que nadie te cuenta',
    brandLogoUrl: 'http://localhost:8000/api/brand-kit/logo',
  },
  cues: [],
});
check(saved !== null, 'la spec guardada se normaliza');
check(saved?.settings.hookText === 'Lo que nadie te cuenta', 'el texto de gancho se conserva');
check(
  saved?.settings.brandLogoUrl === 'http://localhost:8000/api/brand-kit/logo',
  'la URL del logo se conserva (string, no solo null)',
);

// Una spec vieja (sin los campos nuevos) recibe los valores por defecto.
const legacy = normalizeStudioEdit({
  version: 1,
  settings: { ratio: '9:16', trimStart: 0, trimEnd: 12 },
  cues: [],
});
check(legacy !== null, 'la spec vieja se normaliza');
check(legacy?.settings.hookText === '', 'sin gancho, el texto es vacío');
check(legacy?.settings.brandLogoUrl === null, 'sin logo, la URL es null');

// Un logo guardado con un tipo inválido se descarta (no rompe la spec).
const broken = normalizeStudioEdit({
  version: 1,
  settings: { ...DEFAULT_SETTINGS, brandLogoUrl: 42 },
  cues: [],
});
check(broken?.settings.brandLogoUrl === null, 'un logo no string se descarta');

console.log(`\neditor.test.ts: ${checks} comprobaciones superadas.`);
