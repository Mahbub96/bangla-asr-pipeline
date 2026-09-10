export function InputLevelMeter({ level }: { level: number }) {
  return (
    <div className="live-meter" aria-label="Live microphone input level">
      {Array.from({ length: 24 }, (_, index) => {
        const lit = index / 23 < level;
        return <span key={index} className={lit ? "active" : ""} />;
      })}
    </div>
  );
}
