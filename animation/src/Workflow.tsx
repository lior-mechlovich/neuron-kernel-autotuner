import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from "remotion";

const SANS = 'Inter, -apple-system, "Segoe UI", Roboto, sans-serif';
const MONO = 'ui-monospace, "SF Mono", Menlo, monospace';

type Step = {label: string; cap: string; who: "AWS" | "us"; icon: string};
const STEPS: Step[] = [
  {label: "baseline", cap: "profile the kernel on device", who: "AWS", icon: "📈"},
  {label: "find bottleneck", cap: "memory-bound · DMA 16%", who: "AWS", icon: "🔎"},
  {label: "generate", cap: "write a faster variant", who: "AWS", icon: "✍️"},
  {label: "compile ✓", cap: "neuronx-cc compiles it", who: "AWS", icon: "⚙️"},
  {label: "verify ✓", cap: "correct on 3 input sets — hardened gate", who: "us", icon: "🔒"},
  {label: "profile", cap: "neuron-bench · device latency", who: "AWS", icon: "⏱"},
  {label: "rank", cap: "keep the fastest correct variant", who: "us", icon: "🏆"},
];

const LOOP_START = 70;
const STEP_DUR = 38;
const RESULT_START = LOOP_START + STEPS.length * STEP_DUR + 10; // ~346

const Node: React.FC<{s: Step; active: boolean; done: boolean}> = ({s, active, done}) => {
  const isUs = s.who === "us";
  const base = isUs ? "70,224,160" : "127,176,255";
  const bg = active ? `rgba(${base},0.18)` : done ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.02)";
  const border = active ? `rgb(${base})` : `rgba(${base},0.3)`;
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
      width: 140, padding: "12px 6px", borderRadius: 12,
      background: bg, border: `1.5px solid ${border}`,
      transform: active ? "scale(1.06)" : "scale(1)",
      boxShadow: active ? `0 0 24px rgba(${base},0.5)` : "none",
      transition: "all 0.2s",
    }}>
      <div style={{fontSize: 22}}>{s.icon}</div>
      <div style={{fontSize: 16, fontWeight: 700, color: "#e8eefc", textAlign: "center"}}>{s.label}</div>
      <div style={{fontSize: 10, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase",
        color: isUs ? "#46e0a0" : "#7f97bf"}}>{isUs ? "us" : "aws"}</div>
    </div>
  );
};

export const Workflow: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  // intro
  const titleO = interpolate(frame, [0, 25], [0, 1], {extrapolateRight: "clamp"});
  const titleY = interpolate(frame, [0, 25], [20, 0], {extrapolateRight: "clamp"});

  // which step is active
  const rawIdx = Math.floor((frame - LOOP_START) / STEP_DUR);
  const activeIdx = Math.max(-1, Math.min(STEPS.length - 1, rawIdx));
  const inResult = frame >= RESULT_START;

  // result animations
  const rf = frame - RESULT_START;
  const barGrow = interpolate(rf, [10, 45], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});
  const baseUs = Math.round(interpolate(rf, [10, 45], [0, 463], {extrapolateLeft: "clamp", extrapolateRight: "clamp"}));
  const optUs = Math.round(interpolate(rf, [10, 45], [0, 273], {extrapolateLeft: "clamp", extrapolateRight: "clamp"}));
  const xPop = spring({frame: rf - 55, fps, config: {damping: 9, mass: 0.7}});
  const tagO = interpolate(rf, [95, 120], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"});

  const BAR_W = 720;

  return (
    <AbsoluteFill style={{
      fontFamily: SANS,
      background: "radial-gradient(900px 520px at 18% -10%, #1b2a4a 0%, #0b1020 55%, #070a14 100%)",
      color: "#e8edf7",
    }}>
      {/* header */}
      <div style={{position: "absolute", top: 54, left: 64, right: 64, opacity: titleO, transform: `translateY(${titleY}px)`}}>
        <div style={{letterSpacing: 6, fontSize: 15, fontWeight: 600, color: "#7fb0ff", textTransform: "uppercase"}}>
          neuron-kernel-autotuner
        </div>
        <div style={{fontSize: 40, fontWeight: 800, marginTop: 6, letterSpacing: -1}}>
          The auto-tuning loop, on real Trainium
        </div>
      </div>

      {/* loop row */}
      {!inResult && (
        <div style={{position: "absolute", top: 250, left: 0, right: 0, display: "flex",
          justifyContent: "center", alignItems: "center", gap: 8, padding: "0 50px"}}>
          {STEPS.map((s, i) => (
            <div key={s.label} style={{display: "flex", alignItems: "center", gap: 8,
              opacity: interpolate(frame, [LOOP_START + i * 6 - 20, LOOP_START + i * 6], [0.25, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"})}}>
              <Node s={s} active={i === activeIdx} done={i < activeIdx} />
              {i < STEPS.length - 1 && <div style={{color: "#5773a6", fontSize: 22, fontWeight: 800}}>→</div>}
            </div>
          ))}
        </div>
      )}

      {/* active caption */}
      {!inResult && activeIdx >= 0 && (
        <div style={{position: "absolute", top: 400, left: 0, right: 0, textAlign: "center"}}>
          <div style={{fontSize: 30, fontWeight: 700, color: STEPS[activeIdx].who === "us" ? "#7ff0c6" : "#bcd2ff"}}>
            {STEPS[activeIdx].icon}&nbsp; {STEPS[activeIdx].label}
          </div>
          <div style={{fontSize: 22, color: "#9fb0cc", marginTop: 10}}>{STEPS[activeIdx].cap}</div>
        </div>
      )}
      {!inResult && (
        <div style={{position: "absolute", top: 510, left: 0, right: 0, textAlign: "center",
          fontSize: 18, color: "#7f93b8",
          opacity: interpolate(frame, [LOOP_START + 5 * STEP_DUR, LOOP_START + 6 * STEP_DUR], [0, 1], {extrapolateLeft: "clamp", extrapolateRight: "clamp"})}}>
          ↺ beam search · optimization memory &nbsp;—&nbsp; correctness runs <b style={{color: "#cdd9ee"}}>before</b> speed
        </div>
      )}

      {/* result */}
      {inResult && (
        <div style={{position: "absolute", top: 200, left: 0, right: 0, display: "flex",
          flexDirection: "column", alignItems: "center"}}>
          <div style={{fontSize: 18, color: "#8aa0c4", letterSpacing: 1, textTransform: "uppercase"}}>
            Measured on trn1.2xlarge · device latency
          </div>

          {/* baseline bar */}
          <div style={{marginTop: 30, width: BAR_W}}>
            <div style={{display: "flex", justifyContent: "space-between", fontSize: 18, marginBottom: 6}}>
              <span style={{fontFamily: MONO, color: "#cdd9ee"}}>nki_matmul_tiled_</span>
              <span style={{fontFamily: MONO, color: "#ff9d7a", fontWeight: 700}}>{baseUs} µs</span>
            </div>
            <div style={{height: 30, borderRadius: 8, background: "#11182c"}}>
              <div style={{height: "100%", width: `${barGrow * 100}%`, borderRadius: 8,
                background: "linear-gradient(90deg,#c25a3a,#ff9d7a)"}} />
            </div>
          </div>

          {/* optimized bar */}
          <div style={{marginTop: 20, width: BAR_W}}>
            <div style={{display: "flex", justifyContent: "space-between", fontSize: 18, marginBottom: 6}}>
              <span style={{fontFamily: MONO, color: "#cdd9ee"}}>nki_matmul_fully_optimized_</span>
              <span style={{fontFamily: MONO, color: "#46e0a0", fontWeight: 700}}>{optUs} µs ✓</span>
            </div>
            <div style={{height: 30, borderRadius: 8, background: "#11182c"}}>
              <div style={{height: "100%", width: `${barGrow * (273 / 463) * 100}%`, borderRadius: 8,
                background: "linear-gradient(90deg,#2a8f63,#46e0a0)"}} />
            </div>
          </div>

          {/* 1.70x */}
          <div style={{marginTop: 34, transform: `scale(${0.4 + xPop * 0.6})`, opacity: Math.min(1, xPop * 1.2)}}>
            <span style={{fontFamily: MONO, fontWeight: 900, fontSize: 96,
              background: "linear-gradient(90deg,#46e0a0,#7fb0ff)", WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent"}}>1.70×</span>
          </div>
          <div style={{fontSize: 22, color: "#9fb0cc", marginTop: 2, opacity: Math.min(1, xPop)}}>
            faster on the NeuronCore — verified correct
          </div>

          <div style={{marginTop: 30, fontSize: 26, fontWeight: 800, opacity: tagO}}>
            AWS gives you the parts. <span style={{color: "#7fb0ff"}}>This is the loop.</span>
          </div>
        </div>
      )}
    </AbsoluteFill>
  );
};
