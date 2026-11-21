// Render del editor en canvas 1080×1920, adaptado de ShortStudio para
// kellyClipAi v1.3.0. Cambios respecto al original: se eliminó el modo
// demo (el Studio siempre edita el video fuente del proyecto).
import { EditorSettings, SubtitleCue } from '../types/editor';
import { hexToRgba, isColorBright } from './srt';

export const CANVAS_WIDTH = 1080;
export const CANVAS_HEIGHT = 1920;

interface RenderFrameParams {
  ctx: CanvasRenderingContext2D;
  blurCanvas: HTMLCanvasElement;
  blurCtx: CanvasRenderingContext2D;
  video: HTMLVideoElement | null;
  hasMedia: boolean;
  currentTime: number;
  settings: EditorSettings;
  subtitles: SubtitleCue[];
}

interface ParsedWord {
  word: string;
  idx: number;
  width: number;
  emphasized: boolean;
}

function drawRoundedRectPath(
  c: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  const radius = Math.min(r, w / 2, h / 2);
  c.beginPath();
  c.moveTo(x + radius, y);
  c.lineTo(x + w - radius, y);
  c.quadraticCurveTo(x + w, y, x + w, y + radius);
  c.lineTo(x + w, y + h - radius);
  c.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  c.lineTo(x + radius, y + h);
  c.quadraticCurveTo(x, y + h, x, y + h - radius);
  c.lineTo(x, y + radius);
  c.quadraticCurveTo(x, y, x + radius, y);
  c.closePath();
}

function tokenizeWords(rawText: string): Array<{ word: string; emphasized: boolean }> {
  // Supports multi-word *emphasis spans* and single *word* tokens
  const tokens: Array<{ word: string; emphasized: boolean }> = [];
  const regex = /(\*[^*]+\*|\S+)/g;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(rawText)) !== null) {
    const chunk = match[1];
    if (chunk.startsWith('*') && chunk.endsWith('*') && chunk.length > 2) {
      const inner = chunk.slice(1, -1).trim();
      inner.split(/\s+/).forEach((w) => {
        if (w) tokens.push({ word: w, emphasized: true });
      });
    } else {
      tokens.push({ word: chunk, emphasized: false });
    }
  }
  return tokens;
}

function buildBalancedLines(
  items: ParsedWord[],
  maxLineWidth: number,
  spaceWidth: number
): Array<{ items: ParsedWord[]; width: number }> {
  const totalSingleWidth =
    items.reduce((acc, item) => acc + item.width, 0) +
    Math.max(0, items.length - 1) * spaceWidth;

  if (totalSingleWidth <= maxLineWidth || items.length <= 1) {
    return [{ items, width: totalSingleWidth }];
  }

  // Try balanced 2-line split first if total width fits in ~2 lines
  if (totalSingleWidth <= maxLineWidth * 1.92) {
    let bestSplit = -1;
    let bestDiff = Infinity;
    for (let split = 1; split < items.length; split++) {
      const left = items.slice(0, split);
      const right = items.slice(split);
      const wLeft =
        left.reduce((a, b) => a + b.width, 0) + Math.max(0, left.length - 1) * spaceWidth;
      const wRight =
        right.reduce((a, b) => a + b.width, 0) + Math.max(0, right.length - 1) * spaceWidth;
      if (wLeft <= maxLineWidth && wRight <= maxLineWidth) {
        const diff = Math.abs(wLeft - wRight);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestSplit = split;
        }
      }
    }
    if (bestSplit !== -1) {
      const left = items.slice(0, bestSplit);
      const right = items.slice(bestSplit);
      const wLeft =
        left.reduce((a, b) => a + b.width, 0) + Math.max(0, left.length - 1) * spaceWidth;
      const wRight =
        right.reduce((a, b) => a + b.width, 0) + Math.max(0, right.length - 1) * spaceWidth;
      return [
        { items: left, width: wLeft },
        { items: right, width: wRight },
      ];
    }
  }

  // Fallback greedy wrapping for 3+ lines
  const lines: Array<{ items: ParsedWord[]; width: number }> = [];
  let currentLine: ParsedWord[] = [];
  let currentWidth = 0;

  items.forEach((item) => {
    const addW = currentLine.length === 0 ? item.width : spaceWidth + item.width;
    if (currentLine.length > 0 && currentWidth + addW > maxLineWidth) {
      lines.push({ items: currentLine, width: currentWidth });
      currentLine = [item];
      currentWidth = item.width;
    } else {
      currentLine.push(item);
      currentWidth += addW;
    }
  });
  if (currentLine.length > 0) {
    lines.push({ items: currentLine, width: currentWidth });
  }
  return lines;
}

function drawSubtitles(
  ctx: CanvasRenderingContext2D,
  t: number,
  settings: EditorSettings,
  subtitles: SubtitleCue[]
) {
  const activeSub = subtitles.find((s) => t >= s.start && t <= s.end);
  if (!activeSub || !activeSub.text.trim()) return;

  const rawText = settings.subUpper ? activeSub.text.toUpperCase() : activeSub.text;
  const tokens = tokenizeWords(rawText);
  if (tokens.length === 0) return;

  ctx.save();
  ctx.font = `900 ${settings.subSize}px ${settings.subFont}`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  ctx.lineJoin = 'round';
  ctx.miterLimit = 2;

  // Compute weighted active word index + intra-word progress
  let activeWordIdx = -1;
  let wordProgress = 1;
  if (settings.subMode !== 'classic') {
    const dur = Math.max(0.05, activeSub.end - activeSub.start);
    const progress = Math.max(0, Math.min(0.9999, (t - activeSub.start) / dur));
    const weights = tokens.map((tk) => tk.word.length + 2);
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    let acc = 0;
    for (let i = 0; i < tokens.length; i++) {
      const span = weights[i] / totalWeight;
      if (progress <= acc + span || i === tokens.length - 1) {
        activeWordIdx = i;
        wordProgress = Math.max(0, Math.min(1, (progress - acc) / Math.max(0.001, span)));
        break;
      }
      acc += span;
    }
  }

  const parsedWords: ParsedWord[] = tokens.map((tk, idx) => ({
    word: tk.word,
    emphasized: tk.emphasized,
    idx,
    width: ctx.measureText(tk.word).width,
  }));

  // In progressive reveal mode, only lay out words spoken up to activeWordIdx
  const visibleWords =
    settings.subMode === 'reveal' && activeWordIdx >= 0
      ? parsedWords.slice(0, activeWordIdx + 1)
      : parsedWords;

  const spaceWidth = ctx.measureText(' ').width * 1.18;
  const lines = buildBalancedLines(visibleWords, settings.subMaxWidth, spaceWidth);

  // Block pop-in animation
  let popScale = 1;
  if (settings.subPop) {
    const elapsed = Math.max(0, t - activeSub.start);
    if (elapsed < 0.14) {
      const p = elapsed / 0.14;
      const c1 = 1.70158;
      const c3 = c1 + 1;
      const eased = 1 + c3 * Math.pow(p - 1, 3) + c1 * Math.pow(p - 1, 2);
      popScale = 0.74 + 0.26 * eased;
    }
  }

  const lineHeight = settings.subSize * 1.26;
  const totalHeight = lines.length * lineHeight;
  const maxContentWidth = Math.max(...lines.map((l) => l.width), 80);
  const centerY = (settings.subY / 100) * CANVAS_HEIGHT;

  ctx.translate(CANVAS_WIDTH / 2, centerY);
  ctx.scale(popScale, popScale);

  // Block-level background container (line-box or glass-card)
  if (settings.subBgMode === 'line-box' || settings.subBgMode === 'glass-card') {
    const padX = 34;
    const padY = 20;
    const boxW = maxContentWidth + padX * 2;
    const boxH = totalHeight + padY * 2 - settings.subSize * 0.18;
    const isGlass = settings.subBgMode === 'glass-card';

    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.45)';
    ctx.shadowBlur = 24;
    ctx.shadowOffsetY = 8;
    ctx.fillStyle = isGlass
      ? hexToRgba(settings.subBgColor, 0.76)
      : settings.subBgColor;
    drawRoundedRectPath(ctx, -boxW / 2, -boxH / 2, boxW, boxH, 24);
    ctx.fill();

    if (isGlass) {
      ctx.shadowColor = 'transparent';
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(255,255,255,0.16)';
      ctx.stroke();
    }
    ctx.restore();
  }

  // Smooth spring curve for active word
  const wordSpring =
    wordProgress < 0.35
      ? Math.sin((wordProgress / 0.35) * (Math.PI / 2))
      : 1 - Math.min(1, (wordProgress - 0.35) / 0.65) * 0.35;

  lines.forEach((line, lineIdx) => {
    const y = -totalHeight / 2 + lineIdx * lineHeight + lineHeight / 2;
    let x = -line.width / 2;

    line.items.forEach((item) => {
      const isActive = item.idx === activeWordIdx;
      const isHighlighted =
        settings.subMode === 'classic' ? item.emphasized : isActive || item.emphasized;

      const useWordPill =
        settings.subBgMode === 'word-pill' &&
        (settings.subMode === 'classic' ? item.emphasized : isActive);

      ctx.save();
      const wordCenterX = x + item.width / 2;
      let wordY = y;

      if (isActive) {
        if (settings.subMode === 'bounce') {
          const jump = -10 * wordSpring;
          const sc = 1 + 0.16 * wordSpring;
          wordY += jump;
          ctx.translate(wordCenterX, wordY);
          ctx.scale(sc, sc);
          ctx.translate(-wordCenterX, -wordY);
        } else if (!useWordPill) {
          const sc = 1 + 0.1 * wordSpring;
          ctx.translate(wordCenterX, y);
          ctx.scale(sc, sc);
          ctx.translate(-wordCenterX, -y);
        }
      }

      // Active-word pill badge
      if (useWordPill) {
        const padX = 16;
        const padY = settings.subSize * 0.54;
        ctx.save();
        ctx.shadowColor = 'rgba(0,0,0,0.45)';
        ctx.shadowBlur = 14;
        ctx.shadowOffsetY = 5;
        ctx.fillStyle = settings.subHighlight;
        drawRoundedRectPath(
          ctx,
          x - padX,
          wordY - padY,
          item.width + padX * 2,
          padY * 2,
          18
        );
        ctx.fill();
        ctx.restore();
      }

      // Shadow
      if (settings.subShadow > 0 && !useWordPill) {
        ctx.shadowColor = 'rgba(0,0,0,0.85)';
        ctx.shadowBlur = settings.subShadow;
        ctx.shadowOffsetY = Math.round(settings.subShadow * 0.35);
      }

      // Outer Stroke
      if (settings.subStroke > 0 && !useWordPill) {
        ctx.lineWidth = settings.subStroke;
        ctx.strokeStyle = settings.subStrokeColor;
        ctx.strokeText(item.word, x, wordY);
      }

      // Text Fill
      ctx.shadowColor = 'transparent';
      if (useWordPill) {
        ctx.fillStyle = isColorBright(settings.subHighlight) ? '#090d16' : '#ffffff';
      } else if (isHighlighted) {
        ctx.fillStyle = settings.subHighlight;
      } else if (settings.subBgMode === 'line-box' && isColorBright(settings.subBgColor)) {
        ctx.fillStyle = '#090d16';
      } else {
        ctx.fillStyle = settings.subColor;
      }

      ctx.fillText(item.word, x, wordY);
      ctx.restore();

      x += item.width + spaceWidth;
    });
  });

  ctx.restore();
}

export function renderEditorFrame({
  ctx,
  blurCanvas,
  blurCtx,
  video,
  hasMedia,
  currentTime,
  settings,
  subtitles,
}: RenderFrameParams): void {
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  if (!hasMedia) return;

  const sw = video?.videoWidth || 1080;
  const sh = video?.videoHeight || 1920;

  const zoom = settings.zoom / 100;
  const panX = settings.panX / 100;
  const panY = settings.panY / 100;
  const colorFilter = `brightness(${settings.bright}%) contrast(${settings.contrast}%) saturate(${settings.sat}%)`;

  const paintMedia = (
    targetCtx: CanvasRenderingContext2D,
    dx: number,
    dy: number,
    dw: number,
    dh: number
  ) => {
    if (video && video.readyState >= 2) {
      targetCtx.drawImage(video, dx, dy, dw, dh);
    }
  };

  if (settings.fitMode === 'blur') {
    const bScale = Math.max(blurCanvas.width / sw, blurCanvas.height / sh);
    const bw = sw * bScale;
    const bh = sh * bScale;
    const bx = (blurCanvas.width - bw) / 2;
    const by = (blurCanvas.height - bh) / 2;

    blurCtx.save();
    blurCtx.filter = 'blur(16px) brightness(0.6) saturate(1.15)';
    paintMedia(blurCtx, bx, by, bw, bh);
    blurCtx.restore();

    ctx.drawImage(blurCanvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  }

  ctx.save();
  ctx.filter = colorFilter;

  if (settings.fitMode === 'cover') {
    const s = Math.max(CANVAS_WIDTH / sw, CANVAS_HEIGHT / sh) * zoom;
    const dw = sw * s;
    const dh = sh * s;
    const maxShiftX = Math.max(0, (dw - CANVAS_WIDTH) / 2);
    const maxShiftY = Math.max(0, (dh - CANVAS_HEIGHT) / 2);
    const dx = (CANVAS_WIDTH - dw) / 2 + panX * (maxShiftX || CANVAS_WIDTH * 0.15);
    const dy = (CANVAS_HEIGHT - dh) / 2 + panY * (maxShiftY || CANVAS_HEIGHT * 0.15);
    paintMedia(ctx, dx, dy, dw, dh);
  } else {
    const s = Math.min(CANVAS_WIDTH / sw, CANVAS_HEIGHT / sh) * zoom;
    const dw = sw * s;
    const dh = sh * s;
    const dx = (CANVAS_WIDTH - dw) / 2 + panX * CANVAS_WIDTH * 0.25;
    const dy = (CANVAS_HEIGHT - dh) / 2 + panY * CANVAS_HEIGHT * 0.25;

    if (settings.fitMode === 'blur' && settings.rounded) {
      ctx.save();
      ctx.shadowColor = 'rgba(0,0,0,0.65)';
      ctx.shadowBlur = 45;
      ctx.shadowOffsetY = 16;
      drawRoundedRectPath(ctx, dx, dy, dw, dh, 38);
      ctx.fillStyle = '#000';
      ctx.fill();
      ctx.restore();

      ctx.save();
      drawRoundedRectPath(ctx, dx, dy, dw, dh, 38);
      ctx.clip();
      paintMedia(ctx, dx, dy, dw, dh);
      ctx.restore();
    } else {
      paintMedia(ctx, dx, dy, dw, dh);
    }
  }
  ctx.restore();

  if (settings.progressBar) {
    const segDur = Math.max(0.1, settings.trimEnd - settings.trimStart);
    const progress = Math.max(0, Math.min(1, (currentTime - settings.trimStart) / segDur));
    const barH = 14;
    const barY = settings.progressPos === 'top' ? 0 : CANVAS_HEIGHT - barH;
    ctx.fillStyle = 'rgba(255,255,255,0.22)';
    ctx.fillRect(0, barY, CANVAS_WIDTH, barH);
    ctx.fillStyle = settings.progressColor;
    ctx.fillRect(0, barY, CANVAS_WIDTH * progress, barH);
  }

  drawSubtitles(ctx, currentTime, settings, subtitles);
}
