import type { RiskLevel } from "./utils/colorScale";

export interface Hexagon {
  h3: string;
  lat: number;
  lon: number;
  probability: number;
  risk: RiskLevel;
  temperature: number;
  humidity: number;
  wind_speed: number;
  precipitation: number;
  active_fires: number;
  region?: string;
}

export interface HexResponse {
  resolution: number;
  updated_at: string;
  hexagons: Hexagon[];
}

export interface StatusResponse {
  firms_updated: string;
  weather_updated: string;
  model_loaded: boolean;
  alert_count: number;
  hex_count: number;
}

export type BoundaryGeoJSON = {
  type: "FeatureCollection";
  features: unknown[];
};
