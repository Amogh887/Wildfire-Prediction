import { probabilityToColor, rgbaCss } from "../utils/colorScale";

const SAMPLES = 24;
const gradient = `linear-gradient(to right, ${Array.from(
  { length: SAMPLES },
  (_, i) => {
    const p = i / (SAMPLES - 1);
    return `${rgbaCss(probabilityToColor(p))} ${(p * 100).toFixed(0)}%`;
  }
).join(", ")})`;

const TICKS = ["LOW", "MODERATE", "HIGH", "EXTREME"];

export default function Legend() {
  return (
    <div className="legend">
      <div className="legend-title">WILDFIRE PROBABILITY</div>
      <div className="legend-bar" style={{ background: gradient }} />
      <div className="legend-ticks">
        {TICKS.map((t) => (
          <span key={t}>{t}</span>
        ))}
      </div>
    </div>
  );
}
