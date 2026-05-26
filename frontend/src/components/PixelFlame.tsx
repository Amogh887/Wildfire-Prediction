interface Props {
  x: number;
  y: number;
  delay?: number;
}

// 8-col × 12-row pixel art flame grid.
// 0=transparent 1=dark-red 2=red-orange 3=orange 4=yellow
const GRID = [
  [0, 0, 0, 1, 1, 0, 0, 0],
  [0, 0, 1, 2, 1, 1, 0, 0],
  [0, 1, 1, 2, 2, 1, 1, 0],
  [0, 1, 2, 2, 3, 2, 1, 0],
  [1, 1, 2, 3, 3, 2, 1, 1],
  [1, 2, 2, 3, 3, 3, 2, 1],
  [1, 2, 3, 3, 4, 3, 2, 1],
  [1, 2, 3, 4, 4, 3, 3, 1],
  [2, 3, 3, 4, 4, 4, 3, 2],
  [2, 3, 4, 4, 4, 4, 3, 2],
  [3, 3, 4, 4, 4, 4, 3, 3],
  [3, 4, 4, 4, 4, 4, 4, 3],
];

const COLORS = ["transparent", "#c42000", "#f04800", "#ff7c00", "#ffc800"];
const COLS = 8;
const PX = 2; // CSS pixels per flame pixel (half of original 3)

export default function PixelFlame({ x, y, delay = 0 }: Props) {
  const w = COLS * PX;
  const h = GRID.length * PX;
  return (
    <div
      className="pixel-flame"
      style={{
        left: x - w / 2,
        top: y - h,
        width: w,
        height: h,
        gridTemplateColumns: `repeat(${COLS}, ${PX}px)`,
        animationDelay: `${delay}ms`,
      }}
    >
      {GRID.flat().map((v, i) => (
        <span key={i} style={{ background: COLORS[v] }} />
      ))}
    </div>
  );
}
