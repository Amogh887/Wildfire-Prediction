import type { ConnState } from "../hooks/useHexData";
import type { StatusResponse } from "../types";

interface Props {
  status: StatusResponse | null;
  updatedAt: string | null;
  conn: ConnState;
}

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "short",
  });
}

const CONN_LABEL: Record<ConnState, string> = {
  connecting: "CONNECTING TO BACKEND…",
  connected: "LIVE",
  error: "BACKEND UNREACHABLE",
};

export default function Header({ status, updatedAt, conn }: Props) {
  return (
    <header className="app-header">
      <div className="brand">
        <span className="brand-mark" />
        <div className="brand-text">
          <h1>PYROSCOPE</h1>
          <p>Wildfire Watch — Australia</p>
        </div>
      </div>

      <div className="header-stats">
        <div className="stat">
          <span className="stat-label">ACTIVE ALERTS</span>
          <span className="stat-value alert-count">
            {status ? status.alert_count : "—"}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">HEXES</span>
          <span className="stat-value">
            {status ? status.hex_count.toLocaleString() : "—"}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">UPDATED</span>
          <span className="stat-value">{formatTime(updatedAt)}</span>
        </div>
        <div className={`conn conn-${conn}`}>
          <span className="conn-dot" />
          {CONN_LABEL[conn]}
        </div>
      </div>
    </header>
  );
}
