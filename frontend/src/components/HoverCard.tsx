import type { Hexagon } from "../types";
import { riskColorCss } from "../utils/colorScale";

interface Props {
  hex: Hexagon;
  x: number;
  y: number;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="hc-row">
      <span className="hc-label">{label}</span>
      <span className="hc-value">{value}</span>
    </div>
  );
}

export default function HoverCard({ hex, x, y }: Props) {
  // Flip the card to the other side of the cursor near the viewport edge.
  const flipX = x > window.innerWidth - 280;
  const flipY = y > window.innerHeight - 240;
  const style: React.CSSProperties = {
    left: flipX ? undefined : x + 16,
    right: flipX ? window.innerWidth - x + 16 : undefined,
    top: flipY ? undefined : y + 16,
    bottom: flipY ? window.innerHeight - y + 16 : undefined,
  };

  const region = hex.region ?? `${hex.lat.toFixed(2)}, ${hex.lon.toFixed(2)}`;

  return (
    <div className="hover-card" style={style}>
      <div className="hc-head">
        <span className="hc-region">{region}</span>
        <span
          className="hc-badge"
          style={{
            color: riskColorCss(hex.risk),
            borderColor: riskColorCss(hex.risk),
          }}
        >
          {hex.risk}
        </span>
      </div>
      <div className="hc-prob">
        {(hex.probability * 100).toFixed(0)}
        <span className="hc-prob-unit">% probability</span>
      </div>
      <div className="hc-grid">
        <Row label="Temp" value={`${hex.temperature.toFixed(1)} C`} />
        <Row label="Humidity" value={`${hex.humidity.toFixed(0)} %`} />
        <Row label="Wind" value={`${hex.wind_speed.toFixed(0)} km/h`} />
        <Row label="Precip" value={`${hex.precipitation.toFixed(1)} mm`} />
      </div>
      <div className="hc-fires">
        Active fires nearby:{" "}
        <span className={hex.active_fires > 0 ? "hc-fires-on" : ""}>
          {hex.active_fires}
        </span>
      </div>
    </div>
  );
}
