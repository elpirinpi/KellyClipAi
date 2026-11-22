// Puntuación de viralidad de un clip (v1.4.3): los clips manuales
// (exportados desde el Studio sin clip_id) no tienen puntuación — el
// backend los guarda con score null, y eso es honesto: un clip hecho
// a mano no pasó por el scorer. Mostrar y ordenar la puntuación debe
// tolerar nulls: antes, ResultsPage hacía clip.score.toFixed(1) sin
// protección y un solo clip manual tumbaba toda la interfaz con
// "Algo salió mal" al volver a la lista de clips.

export type ScoredClip = { score: number | null };

/** Puntuación lista para mostrar: un decimal, o "—" sin puntuación. */
export function formatClipScore(score: number | null | undefined): string {
  if (typeof score !== 'number' || !Number.isFinite(score)) return '—';
  return score.toFixed(1);
}

/** Ordena por puntuación descendente, con los clips sin puntuación al final. */
export function compareClipsByScore(a: ScoredClip, b: ScoredClip): number {
  const aScore = a.score;
  const bScore = b.score;
  if (typeof aScore === 'number' && Number.isFinite(aScore)) {
    if (typeof bScore === 'number' && Number.isFinite(bScore)) return bScore - aScore;
    return -1;
  }
  if (typeof bScore === 'number' && Number.isFinite(bScore)) return 1;
  return 0;
}
