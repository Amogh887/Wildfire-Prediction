import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { MapView, WebMercatorViewport } from "@deck.gl/core";
import type { PickingInfo, MapViewState } from "@deck.gl/core";
import { GeoJsonLayer, ScatterplotLayer, TextLayer, PathLayer } from "@deck.gl/layers";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { MaskExtension } from "@deck.gl/extensions";
import { cellToLatLng } from "h3-js";
import type { Hexagon, HexResponse, BoundaryGeoJSON } from "../types";
import { probabilityToColor } from "../utils/colorScale";
import PixelFlame from "./PixelFlame";

interface Props {
  hexData: HexResponse | null;
  boundary: BoundaryGeoJSON | null;
  onHover: (hex: Hexagon | null, x: number, y: number) => void;
}

const INITIAL_VIEW_STATE: MapViewState = {
  longitude: 134,
  latitude: -27.5,
  zoom: 3.4,
  minZoom: 2.6,
  maxZoom: 9,
  pitch: 0,
  bearing: 0,
};

// ── Geographic annotation data ─────────────────────────────────────────
interface GeoPoint { name: string; lon: number; lat: number }
interface RiverLine { name: string; coords: [number, number][] }

const CITIES: GeoPoint[] = [
  { name: "SYDNEY",    lon: 151.209, lat: -33.868 },
  { name: "MELBOURNE", lon: 144.963, lat: -37.814 },
  { name: "BRISBANE",  lon: 153.025, lat: -27.470 },
  { name: "PERTH",     lon: 115.861, lat: -31.952 },
  { name: "ADELAIDE",  lon: 138.601, lat: -34.928 },
  { name: "DARWIN",    lon: 130.845, lat: -12.462 },
  { name: "HOBART",    lon: 147.328, lat: -42.882 },
  { name: "CANBERRA",  lon: 149.128, lat: -35.282 },
];

const SEAS: GeoPoint[] = [
  { name: "CORAL SEA",              lon: 155.0, lat: -16.0 },
  { name: "TASMAN SEA",             lon: 158.5, lat: -38.0 },
  { name: "INDIAN OCEAN",           lon: 107.0, lat: -28.5 },
  { name: "SOUTHERN OCEAN",         lon: 132.0, lat: -48.5 },
  { name: "TIMOR SEA",              lon: 128.5, lat: -11.2 },
  { name: "ARAFURA SEA",            lon: 135.5, lat:  -9.0 },
  { name: "GREAT AUSTRALIAN BIGHT", lon: 126.0, lat: -34.5 },
];

const RIVERS: RiverLine[] = [
  {
    name: "Murray",
    coords: [
      [148.2, -36.5], [146.92, -36.07], [143.55, -35.34],
      [142.15, -34.18], [140.75, -34.17], [139.28, -35.22],
    ],
  },
  {
    name: "Darling",
    coords: [
      [145.94, -29.50], [145.72, -30.68],
      [143.73, -31.95], [142.40, -32.40], [141.92, -34.10],
    ],
  },
  {
    name: "Murrumbidgee",
    coords: [
      [149.14, -35.47], [147.37, -35.12], [146.56, -34.75],
      [144.50, -34.50], [143.53, -34.65], [143.02, -34.62],
    ],
  },
];

export default function WildfireScene({ hexData, boundary, onHover }: Props) {
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);
  const [canvasSize, setCanvasSize] = useState({ w: window.innerWidth, h: window.innerHeight });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        setCanvasSize({ w: e.contentRect.width, h: e.contentRect.height });
      }
    });
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const handleViewStateChange = useCallback(
    (params: { viewState: MapViewState }) => setViewState(params.viewState),
    []
  );

  const view = useMemo(() => new MapView({ id: "map", controller: true }), []);
  const hexagons = hexData?.hexagons ?? [];

  // Screen positions for fire-hex flame overlays.
  const flamePositions = useMemo(() => {
    const fireHexes = hexagons.filter((h) => h.active_fires > 0);
    if (!fireHexes.length) return [];
    const vp = new WebMercatorViewport({
      width: canvasSize.w,
      height: canvasSize.h,
      longitude: viewState.longitude,
      latitude: viewState.latitude,
      zoom: viewState.zoom,
      pitch: viewState.pitch ?? 0,
      bearing: viewState.bearing ?? 0,
    });
    return fireHexes.map((h) => {
      const [lat, lng] = cellToLatLng(h.h3);
      const [x, y] = vp.project([lng, lat]);
      const delay = Math.round((x * 37 + y * 53) % 350);
      return { h3: h.h3, x, y, delay };
    });
  }, [hexagons, viewState, canvasSize]);

  const layers = useMemo(() => {
    const result = [];

    // Mask: clips hexes to Australia coastline.
    if (boundary) {
      result.push(
        new GeoJsonLayer({
          id: "australia-mask",
          data: boundary as never,
          operation: "mask",
          filled: true,
          stroked: false,
        })
      );
    }

    result.push(
      new H3HexagonLayer<Hexagon>({
        id: "hexagons",
        data: hexagons,
        pickable: true,
        filled: true,
        stroked: true,
        extruded: false,
        highPrecision: true,
        getHexagon: (d) => d.h3,
        getFillColor: (d) => probabilityToColor(d.probability),
        getLineColor: (d) =>
          d.probability < 0.05 ? [255, 255, 255, 30] : [255, 255, 255, 55],
        lineWidthUnits: "pixels",
        getLineWidth: 0.5,
        coverage: 0.9,
        extensions: [new MaskExtension()],
        ...({ maskId: "australia-mask", maskByInstance: false } as Record<
          string,
          unknown
        >),
        updateTriggers: {
          getFillColor: hexData?.updated_at,
          getLineColor: hexData?.updated_at,
        },
      })
    );

    // White coastline above hexes.
    if (boundary) {
      result.push(
        new GeoJsonLayer({
          id: "australia-outline",
          data: boundary as never,
          stroked: true,
          filled: false,
          getLineColor: [255, 255, 255, 230],
          lineWidthUnits: "pixels",
          getLineWidth: 1.2,
          lineWidthMinPixels: 1,
          pickable: false,
        })
      );
    }

    // Rivers — thin blue-white lines.
    result.push(
      new PathLayer<RiverLine>({
        id: "rivers",
        data: RIVERS,
        pickable: false,
        getPath: (d) => d.coords as [number, number][],
        getColor: [160, 200, 255, 65],
        getWidth: 1,
        widthUnits: "pixels",
        widthMinPixels: 0.8,
      })
    );

    // City dots.
    result.push(
      new ScatterplotLayer<GeoPoint>({
        id: "city-dots",
        data: CITIES,
        pickable: false,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: 2.5,
        radiusUnits: "pixels",
        getFillColor: [255, 255, 255, 220],
        stroked: false,
      })
    );

    // City labels — monospace, appear above each dot.
    result.push(
      new TextLayer<GeoPoint>({
        id: "city-labels",
        data: CITIES,
        pickable: false,
        getPosition: (d) => [d.lon, d.lat],
        getText: (d) => d.name,
        getSize: 10,
        getColor: [255, 255, 255, 190],
        getPixelOffset: [0, -9],
        getTextAnchor: "middle",
        getAlignmentBaseline: "bottom",
        fontFamily: "monospace",
        fontWeight: 600,
        sizeScale: 1,
      })
    );

    // Sea / ocean labels — larger, dimmer.
    result.push(
      new TextLayer<GeoPoint>({
        id: "sea-labels",
        data: SEAS,
        pickable: false,
        getPosition: (d) => [d.lon, d.lat],
        getText: (d) => d.name,
        getSize: 11,
        getColor: [170, 205, 255, 75],
        getTextAnchor: "middle",
        getAlignmentBaseline: "center",
        fontFamily: "monospace",
        fontWeight: 300,
        sizeScale: 1,
      })
    );

    return result;
  }, [boundary, hexagons, hexData?.updated_at]);

  const handleHover = useCallback(
    (info: PickingInfo) => {
      const obj = info.object as Hexagon | undefined;
      if (obj && info.x != null && info.y != null) onHover(obj, info.x, info.y);
      else onHover(null, 0, 0);
    },
    [onHover]
  );

  return (
    <div
      ref={containerRef}
      style={{ position: "absolute", inset: 0, background: "#000" }}
    >
      <DeckGL
        views={view}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        controller={true}
        layers={layers}
        onHover={handleHover}
        style={{ position: "absolute", inset: "0" }}
      />
      {flamePositions.length > 0 && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            zIndex: 5,
            overflow: "hidden",
          }}
        >
          {flamePositions.map(({ h3, x, y, delay }) => (
            <PixelFlame key={h3} x={x} y={y} delay={delay} />
          ))}
        </div>
      )}
    </div>
  );
}
