// Pruebas de la puntuación de clip a prueba de null (v1.4.3).
//
// Regresión del crash que Kelly reportó el 22-11-2026: exportar un
// clip manual en el Studio y volver a la lista de clips mostraba
// "Algo salió mal" porque el clip manual tiene score null y la lista
// hacía clip.score.toFixed(1) sin protección.
//
// Sin navegador y sin dependencias: se ejecuta con
//   node src/utils/clipScore.test.ts
// (desde app/; Node 22.18+ ejecuta TypeScript directamente) o con
//   npm run test:clip-score
// Cada comprobación que falla lanza y el proceso sale con código 1.
// El archivo no se importa desde la app, así que no entra en el
// bundle; `tsc --noEmit` (npm run build) sí lo comprueba.
import { compareClipsByScore, formatClipScore, type ScoredClip } from './clipScore.ts';

let passed = 0;

function check(name: string, actual: unknown, expected: unknown): void {
  if (actual !== expected) {
    throw new Error(`FAIL: ${name} — esperado ${String(expected)}, obtenido ${String(actual)}`);
  }
  passed += 1;
}

function checkOrder(name: string, actual: ScoredClip[], expected: (number | null)[]): void {
  const scores = actual.map((clip) => clip.score);
  if (scores.length !== expected.length || scores.some((score, index) => score !== expected[index])) {
    throw new Error(`FAIL: ${name} — esperado [${expected.join(', ')}], obtenido [${scores.join(', ')}]`);
  }
  passed += 1;
}

// --- Mostrar la puntuación -------------------------------------------------

check('una puntuación normal se muestra con un decimal', formatClipScore(7.5), '7.5');
check('una puntuación se redondea a un decimal', formatClipScore(7.54), '7.5');
check('cero es una puntuación válida', formatClipScore(0), '0.0');
check('null se muestra como guion', formatClipScore(null), '—');
check('undefined se muestra como guion', formatClipScore(undefined), '—');
check('NaN se muestra como guion', formatClipScore(NaN), '—');
check('infinito se muestra como guion', formatClipScore(Infinity), '—');

// --- Ordenar por puntuación ------------------------------------------------

check(
  'mayor puntuación primero',
  compareClipsByScore({ score: 8.2 }, { score: 7.5 }) < 0,
  true,
);
check(
  'menor puntuación después',
  compareClipsByScore({ score: 7.5 }, { score: 8.2 }) > 0,
  true,
);
check('puntuaciones iguales empatan', compareClipsByScore({ score: 7.5 }, { score: 7.5 }), 0);
check(
  'un clip sin puntuación va después de uno con puntuación',
  compareClipsByScore({ score: null }, { score: 7.5 }) > 0,
  true,
);
check(
  'un clip con puntuación va antes de uno sin puntuación',
  compareClipsByScore({ score: 7.5 }, { score: null }) < 0,
  true,
);
check('dos clips sin puntuación empatan', compareClipsByScore({ score: null }, { score: null }), 0);

const mixed: ScoredClip[] = [
  { score: null },
  { score: 7.5 },
  { score: null },
  { score: 9.1 },
  { score: 8.2 },
];
checkOrder(
  'ordenar mezcla puntuados descendente y nulls al final',
  [...mixed].sort(compareClipsByScore),
  [9.1, 8.2, 7.5, null, null],
);

console.log(`clipScore.test.ts: ${passed} comprobaciones superadas.`);
