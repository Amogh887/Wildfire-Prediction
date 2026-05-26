import { useCallback, useEffect, useRef, useState } from "react";
import type {
  BoundaryGeoJSON,
  HexResponse,
  StatusResponse,
} from "../types";

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  "http://localhost:8000";

const POLL_MS = 60_000;

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return (await res.json()) as T;
}

export type ConnState = "connecting" | "connected" | "error";

export interface HexDataResult {
  hexData: HexResponse | null;
  status: StatusResponse | null;
  boundary: BoundaryGeoJSON | null;
  conn: ConnState;
  errorMsg: string | null;
  resolution: number;
  setResolution: (r: number) => void;
  refresh: () => void;
}

export function useHexData(initialResolution = 4): HexDataResult {
  const [resolution, setResolution] = useState(initialResolution);
  const [hexData, setHexData] = useState<HexResponse | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [boundary, setBoundary] = useState<BoundaryGeoJSON | null>(null);
  const [conn, setConn] = useState<ConnState>("connecting");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Keep latest resolution in a ref so the poll loop reads fresh value.
  const resRef = useRef(resolution);
  resRef.current = resolution;
  const hasDataRef = useRef(false);

  const fetchAll = useCallback(async (signal?: AbortSignal) => {
    const r = resRef.current;
    try {
      const [hex, st] = await Promise.all([
        getJson<HexResponse>(`/api/hexagons?resolution=${r}`, signal),
        getJson<StatusResponse>(`/api/status`, signal),
      ]);
      if (signal?.aborted) return;
      setHexData(hex);
      setStatus(st);
      hasDataRef.current = true;
      setConn("connected");
      setErrorMsg(null);
    } catch (err) {
      if (signal?.aborted || (err as Error).name === "AbortError") return;
      // Only flip to error UI when we have never received data.
      setConn(hasDataRef.current ? "connected" : "error");
      setErrorMsg((err as Error).message);
    }
  }, []);

  // Boundary fetched once (retried if missing).
  useEffect(() => {
    const ctrl = new AbortController();
    let alive = true;
    const loadBoundary = async () => {
      try {
        const b = await getJson<BoundaryGeoJSON>(`/api/boundary`, ctrl.signal);
        if (alive) setBoundary(b);
      } catch {
        /* outline is optional; ignore */
      }
    };
    loadBoundary();
    const id = window.setInterval(() => {
      if (!boundary) loadBoundary();
    }, POLL_MS);
    return () => {
      alive = false;
      ctrl.abort();
      window.clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refetch hex+status on resolution change and poll on an interval.
  useEffect(() => {
    const ctrl = new AbortController();
    fetchAll(ctrl.signal);
    const id = window.setInterval(() => fetchAll(), POLL_MS);
    return () => {
      ctrl.abort();
      window.clearInterval(id);
    };
  }, [resolution, fetchAll]);

  const refresh = useCallback(() => {
    fetchAll();
  }, [fetchAll]);

  return {
    hexData,
    status,
    boundary,
    conn,
    errorMsg,
    resolution,
    setResolution,
    refresh,
  };
}
