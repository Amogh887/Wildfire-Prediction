import { useEffect } from "react";
import type { Hexagon } from "../types";
import { riskColorCss } from "../utils/colorScale";

const CITIES = [
  { name: "Sydney", lon: 151.209, lat: -33.868 },
  { name: "Melbourne", lon: 144.963, lat: -37.814 },
  { name: "Brisbane", lon: 153.025, lat: -27.470 },
  { name: "Perth", lon: 115.861, lat: -31.952 },
  { name: "Adelaide", lon: 138.601, lat: -34.928 },
  { name: "Darwin", lon: 130.845, lat: -12.462 },
  { name: "Hobart", lon: 147.328, lat: -42.882 },
  { name: "Canberra", lon: 149.128, lat: -35.282 },
  { name: "Cairns", lon: 145.770, lat: -16.924 },
  { name: "Townsville", lon: 146.818, lat: -19.258 },
  { name: "Alice Springs", lon: 133.882, lat: -23.698 },
  { name: "Broome", lon: 122.236, lat: -17.956 },
  { name: "Port Hedland", lon: 118.594, lat: -20.311 },
];

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.asin(Math.sqrt(a));
}

function nearestCity(lat: number, lon: number): { name: string; km: number } {
  let best = CITIES[0];
  let bestDist = haversineKm(lat, lon, best.lat, best.lon);
  for (const c of CITIES.slice(1)) {
    const d = haversineKm(lat, lon, c.lat, c.lon);
    if (d < bestDist) {
      best = c;
      bestDist = d;
    }
  }
  return { name: best.name, km: Math.round(bestDist) };
}

function stateAbbr(lat: number, lon: number): string {
  if (lat >= -35.95 && lat <= -35.1 && lon >= 148.7 && lon <= 149.5) return "ACT";
  if (lat <= -39.0 && lon >= 143.5 && lon <= 149.0) return "TAS";
  if (lon < 129.0) return "WA";
  if (lon < 138.0) return lat > -26.0 ? "NT" : "SA";
  if (lon < 141.0) return lat < -26.0 ? "SA" : "NT";
  if (lat > -29.0) return "QLD";
  if (lat > -34.0) return "NSW";
  return "VIC";
}

const STATE_FIRE_LINKS: Record<string, Array<{ label: string; url: string }>> = {
  NSW: [{ label: "NSW RFS — Fires Near Me", url: "https://www.rfs.nsw.gov.au/fire-information/fires-near-me" }],
  VIC: [{ label: "VIC Emergency — Fire Map", url: "https://www.emergency.vic.gov.au/respond/" }],
  QLD: [{ label: "QLD Fire — Current Incidents", url: "https://www.fire.qld.gov.au/firemap/" }],
  SA:  [{ label: "CFS South Australia", url: "https://www.cfs.sa.gov.au/live-incidents/" }],
  WA:  [{ label: "WA Emergency — Fire Map", url: "https://www.emergency.wa.gov.au/#/map" }],
  NT:  [{ label: "NT Emergency Alerts", url: "https://www.nt.gov.au/emergency" }],
  TAS: [{ label: "TFS — Current Bushfire List", url: "https://www.fire.tas.gov.au/Show?pageId=colCurrentBushfireList" }],
  ACT: [{ label: "ACT Emergency Services", url: "https://esa.act.gov.au/esa-for-community/alerts-and-warnings" }],
};

interface Props {
  hex: Hexagon | null;
  onClose: () => void;
}

export default function SidePanel({ hex, onClose }: Props) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const open = hex !== null;
  const city = hex ? nearestCity(hex.lat, hex.lon) : null;
  const state = hex ? stateAbbr(hex.lat, hex.lon) : "NSW";
  const fireLinks = STATE_FIRE_LINKS[state] ?? STATE_FIRE_LINKS["NSW"];
  const regionDisplay = hex?.region ?? (hex ? `${hex.lat.toFixed(2)}, ${hex.lon.toFixed(2)}` : "");
  const showVerify = hex && (hex.active_fires > 0 || hex.probability >= 0.7);

  return (
    <div className={`side-panel${open ? " side-panel--open" : ""}`}>
      <div className="sp-header">
        <div className="sp-location">
          <div className="sp-region">{regionDisplay}</div>
          {city && (
            <div className="sp-city">
              {city.km < 40 ? "Near" : "Nearest:"} {city.name}
              {city.km >= 40 && <span className="sp-city-dist"> ({city.km} km)</span>}
            </div>
          )}
        </div>
        <button className="sp-close" onClick={onClose} aria-label="Close">×</button>
      </div>

      {hex && (
        <>
          <div className="sp-prob-row">
            <div className="sp-prob">
              {(hex.probability * 100).toFixed(0)}
              <span className="sp-prob-unit">%</span>
            </div>
            <span
              className="sp-badge"
              style={{ color: riskColorCss(hex.risk), borderColor: riskColorCss(hex.risk) }}
            >
              {hex.risk}
            </span>
          </div>

          <div className="sp-section-label">CONDITIONS</div>
          <div className="sp-grid">
            <div className="sp-cell">
              <span className="sp-cell-label">TEMP</span>
              <span className="sp-cell-value">{hex.temperature.toFixed(1)}°C</span>
            </div>
            <div className="sp-cell">
              <span className="sp-cell-label">HUMIDITY</span>
              <span className="sp-cell-value">{hex.humidity.toFixed(0)}%</span>
            </div>
            <div className="sp-cell">
              <span className="sp-cell-label">WIND</span>
              <span className="sp-cell-value">{hex.wind_speed.toFixed(0)} km/h</span>
            </div>
            <div className="sp-cell">
              <span className="sp-cell-label">PRECIP</span>
              <span className="sp-cell-value">{hex.precipitation.toFixed(1)} mm</span>
            </div>
          </div>

          <div className="sp-fires-row">
            <span className="sp-section-label" style={{ margin: 0 }}>ACTIVE FIRES NEARBY</span>
            <span className={`sp-fires-count${hex.active_fires > 0 ? " sp-fires-count--on" : ""}`}>
              {hex.active_fires}
            </span>
          </div>

          {showVerify && (
            <div className="sp-verify">
              <div className="sp-section-label">VERIFY SOURCES</div>
              <a
                className="sp-link"
                href="https://www.abc.net.au/emergency/"
                target="_blank"
                rel="noopener noreferrer"
              >
                ABC Emergency ↗
              </a>
              {fireLinks.map((l) => (
                <a
                  key={l.url}
                  className="sp-link"
                  href={l.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {l.label} ↗
                </a>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
