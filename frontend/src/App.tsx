import { useCallback, useState } from "react";
import "./App.css";
import { useHexData } from "./hooks/useHexData";
import WildfireScene from "./components/WildfireScene";
import HoverCard from "./components/HoverCard";
import SidePanel from "./components/SidePanel";
import Legend from "./components/Legend";
import Header from "./components/Header";
import type { Hexagon } from "./types";

export default function App() {
  const { hexData, status, boundary, conn, errorMsg } = useHexData(4);

  const [hover, setHover] = useState<{
    hex: Hexagon;
    x: number;
    y: number;
  } | null>(null);

  const [selectedHex, setSelectedHex] = useState<Hexagon | null>(null);

  const handleHover = useCallback(
    (hex: Hexagon | null, x: number, y: number) => {
      setHover(hex ? { hex, x, y } : null);
    },
    []
  );

  const handleSelect = useCallback((hex: Hexagon | null) => {
    setSelectedHex(hex);
  }, []);

  return (
    <div className="app">
      <div className="vignette" />
      <WildfireScene
        hexData={hexData}
        boundary={boundary}
        onHover={handleHover}
        onSelect={handleSelect}
      />

      <Header
        status={status}
        updatedAt={hexData?.updated_at ?? null}
        conn={conn}
      />

      <Legend />

      <SidePanel hex={selectedHex} onClose={() => setSelectedHex(null)} />

      {hover && !selectedHex && (
        <HoverCard hex={hover.hex} x={hover.x} y={hover.y} />
      )}

      {conn === "connecting" && !hexData && (
        <div className="overlay-banner">
          <span className="spinner" />
          Connecting to backend…
        </div>
      )}

      {conn === "error" && !hexData && (
        <div className="overlay-banner error">
          <div className="err-title">Backend unreachable</div>
          <div className="err-sub">
            Could not reach the wildfire API.
            {errorMsg ? ` (${errorMsg})` : ""} Retrying every 60s.
          </div>
        </div>
      )}

      <div className="hint">
        Scroll / pinch to zoom · drag to pan · hover for quick view · click a hex for details
      </div>
    </div>
  );
}
