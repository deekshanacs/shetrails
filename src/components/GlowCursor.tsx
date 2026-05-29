import { useEffect, useState } from "react";

const GlowCursor = () => {
  const [pos, setPos] = useState({ x: -100, y: -100 });
  const [clicking, setClicking] = useState(false);

  useEffect(() => {
    const move = (e: MouseEvent) => setPos({ x: e.clientX, y: e.clientY });
    const down = () => setClicking(true);
    const up = () => setClicking(false);

    window.addEventListener("mousemove", move);
    window.addEventListener("mousedown", down);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mousedown", down);
      window.removeEventListener("mouseup", up);
    };
  }, []);

  return (
    <>
      <div
        className="pointer-events-none fixed z-[9999] rounded-full transition-transform duration-75"
        style={{
          left: pos.x - 20,
          top: pos.y - 20,
          width: 40,
          height: 40,
          background: "radial-gradient(circle, hsla(270,80%,65%,0.3) 0%, transparent 70%)",
          transform: clicking ? "scale(0.8)" : "scale(1)",
        }}
      />
      <div
        className="pointer-events-none fixed z-[9999] rounded-full"
        style={{
          left: pos.x - 4,
          top: pos.y - 4,
          width: 8,
          height: 8,
          background: "hsl(270,80%,65%)",
          boxShadow: "0 0 10px hsl(270 80% 65% / 0.6)",
        }}
      />
    </>
  );
};

export default GlowCursor;
