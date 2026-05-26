// Probability (0..1) -> RGBA gradient + risk-level helpers.

export type RGBA = [number, number, number, number];

type Stop = { t: number; color: RGBA };

// Gradient stops: near-black -> yellow -> orange -> red.
const STOPS: Stop[] = [
  { t: 0.0, color: [10, 10, 10, 180] },
  { t: 0.25, color: [255, 220, 0, 200] },
  { t: 0.6, color: [255, 120, 0, 220] },
  { t: 1.0, color: [220, 20, 20, 240] },
];

function lerp(a: number, b: number, f: number): number {
  return a + (b - a) * f;
}

function clamp01(v: number): number {
  if (Number.isNaN(v)) return 0;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Linearly interpolate the gradient at probability p (0..1). */
export function probabilityToColor(p: number): RGBA {
  const x = clamp01(p);
  for (let i = 0; i < STOPS.length - 1; i++) {
    const lo = STOPS[i];
    const hi = STOPS[i + 1];
    if (x >= lo.t && x <= hi.t) {
      const span = hi.t - lo.t || 1;
      const f = (x - lo.t) / span;
      return [
        Math.round(lerp(lo.color[0], hi.color[0], f)),
        Math.round(lerp(lo.color[1], hi.color[1], f)),
        Math.round(lerp(lo.color[2], hi.color[2], f)),
        Math.round(lerp(lo.color[3], hi.color[3], f)),
      ];
    }
  }
  return STOPS[STOPS.length - 1].color;
}

export type RiskLevel = "LOW" | "MODERATE" | "HIGH" | "EXTREME";

export function probabilityToRisk(p: number): RiskLevel {
  if (p >= 0.75) return "EXTREME";
  if (p >= 0.5) return "HIGH";
  if (p >= 0.25) return "MODERATE";
  return "LOW";
}

/** A representative solid hex/rgb color for a risk level (for badges/legend). */
export function riskColorCss(risk: RiskLevel): string {
  switch (risk) {
    case "EXTREME":
      return "rgb(220,20,20)";
    case "HIGH":
      return "rgb(255,120,0)";
    case "MODERATE":
      return "rgb(255,220,0)";
    default:
      return "rgb(120,120,120)";
  }
}

export function rgbaCss([r, g, b, a]: RGBA): string {
  return `rgba(${r},${g},${b},${(a / 255).toFixed(3)})`;
}
