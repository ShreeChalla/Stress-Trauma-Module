import { useState, useEffect, useRef } from "react";
import {
  Phone,
  ShieldAlert,
  Radio,
  Activity,
  MessageSquareText,
  Siren,
  MapPin,
  Lock,
  Gauge,
  LayoutGrid,
  History,
  Fingerprint,
  Upload,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Scripted demo timeline. This stands in for the live SVI/acoustic/transcript
// pipeline described in the RSTAM spec. Every 3s (matching the spec's 3s
// sliding window) the dashboard advances one step through a mini call, then
// loops. Nothing here calls a backend — it's a front-end simulation only.
// ---------------------------------------------------------------------------
const CALL_SCRIPT = [
  { speaker: "counselor", text: "NHAA helpline, you're speaking with Divya. You're safe on this line — go ahead.", svi: 8, jitter: 0.3, shimmer: 1.1, hnr: 21, f0: 4 },
  { speaker: "caller", text: "Hello... yes, I wanted to report something that happened yesterday.", svi: 14, jitter: 0.6, shimmer: 1.6, hnr: 19, f0: 7 },
  { speaker: "caller", text: "It was just one slap in the field, I am fine, I don't want trouble.", svi: 22, jitter: 0.8, shimmer: 2.0, hnr: 18, f0: 9 },
  { speaker: "counselor", text: "I hear you. Can you tell me a little about who was involved?", svi: 24, jitter: 0.8, shimmer: 2.0, hnr: 18, f0: 9, flag: "power-imbalance" },
  { speaker: "caller", text: "It was the landlord. In front of the whole street. Everyone saw.", svi: 41, jitter: 1.4, shimmer: 3.1, hnr: 15, f0: 14, flag: "sc/st act" },
  { speaker: "caller", text: "They said if I complain they will not let us draw water again.", svi: 52, jitter: 1.9, shimmer: 3.8, hnr: 13, f0: 18, flag: "coercion" },
  { speaker: "counselor", text: "That's not something you have to accept. You did the right thing calling.", svi: 49, jitter: 1.8, shimmer: 3.6, hnr: 13, f0: 17 },
  { speaker: "caller", text: "There— there is someone outside. I have to be quiet a moment.", svi: 74, jitter: 2.9, shimmer: 5.4, hnr: 9, f0: 26, flag: "acoustic threat" },
  { speaker: "caller", text: "[breathing, no speech]", svi: 88, jitter: 3.6, shimmer: 6.7, hnr: 6, f0: 33, flag: "critical override" },
  { speaker: "counselor", text: "I'm still here. If you can't talk, tap once for hiding, twice for hurt.", svi: 91, jitter: 3.4, shimmer: 6.3, hnr: 6, f0: 32 },
  { speaker: "caller", text: "[single tap detected]", svi: 85, jitter: 3.0, shimmer: 5.6, hnr: 8, f0: 28 },
  { speaker: "counselor", text: "Understood, you're hiding. Help is being routed to your location now.", svi: 78, jitter: 2.6, shimmer: 4.9, hnr: 10, f0: 24 },
  { speaker: "caller", text: "Okay... okay they've gone. I think it's safer now.", svi: 58, jitter: 1.9, shimmer: 3.4, hnr: 14, f0: 17 },
  { speaker: "counselor", text: "Good. Stay on the line with me and we'll go through what happens next.", svi: 44, jitter: 1.3, shimmer: 2.6, hnr: 16, f0: 12 },
  { speaker: "caller", text: "Thank you for staying with me.", svi: 30, jitter: 0.9, shimmer: 1.9, hnr: 18, f0: 8 },
];

// Mock immutable audit trail — every AI recommendation and supervisor
// override gets a block. Hashes below are placeholder strings for demo only.
const AUDIT_LOG = [
  { time: "00:00", actor: "System", action: "Call ingested, DPDP consent recorded", hash: "a1f9c3e0" },
  { time: "00:12", actor: "AI", action: "SVI computed: 24 (Moderate) — power-imbalance flag raised", hash: "d84b7211" },
  { time: "00:15", actor: "AI", action: "SC/ST Act ontology match: public humiliation clause", hash: "5e2a9f0c" },
  { time: "00:24", actor: "AI", action: "Critical override triggered — hard trigger delta_critical applied", hash: "9c113ab4" },
  { time: "00:27", actor: "AI", action: "Stealth mode + DTMF keypad triage activated", hash: "77fbe602" },
  { time: "00:30", actor: "Supervisor", action: "Barge-in acknowledged, call taken over", hash: "2d908e1a" },
  { time: "00:36", actor: "Supervisor", action: "Police dispatch (112) request — pending ratification", hash: "f4a01c3d" },
];

function bandFor(svi) {
  if (svi >= 76) return { name: "Critical", text: "text-red-400", bg: "bg-red-500", border: "border-red-500", ring: "ring-red-500/40", dim: "text-red-300/70" };
  if (svi >= 51) return { name: "High", text: "text-orange-400", bg: "bg-orange-500", border: "border-orange-500", ring: "ring-orange-500/40", dim: "text-orange-300/70" };
  if (svi >= 26) return { name: "Moderate", text: "text-amber-400", bg: "bg-amber-500", border: "border-amber-500", ring: "ring-amber-500/40", dim: "text-amber-300/70" };
  return { name: "Low", text: "text-emerald-400", bg: "bg-emerald-500", border: "border-emerald-500", ring: "ring-emerald-500/40", dim: "text-emerald-300/70" };
}

function useElapsed() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  const m = String(Math.floor(seconds / 60)).padStart(2, "0");
  const s = String(seconds % 60).padStart(2, "0");
  return `${m}:${s}`;
}

// EMA smoothing per spec (lambda ~0.3) so the gauge doesn't jump instantly
function useSmoothedSVI(target) {
  const [value, setValue] = useState(target);
  const ref = useRef(target);
  useEffect(() => {
    const id = setInterval(() => {
      ref.current = ref.current + 0.3 * (target - ref.current);
      setValue(ref.current);
    }, 200);
    return () => clearInterval(id);
  }, [target]);
  return value;
}

function SVIGauge({ svi }) {
  const smoothed = useSmoothedSVI(svi);
  const band = bandFor(smoothed);
  const pct = Math.max(0, Math.min(100, smoothed));
  // semicircle arc geometry
  const r = 80;
  const circumference = Math.PI * r;
  const offset = circumference * (1 - pct / 100);

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 120" className="w-full max-w-[170px]">
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="currentColor"
          className="text-slate-800"
          strokeWidth="14"
          strokeLinecap="round"
        />
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="currentColor"
          className={band.text}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 200ms linear" }}
        />
        <text x="100" y="88" textAnchor="middle" className={`font-mono text-2xl fill-current ${band.text}`}>
          {Math.round(smoothed)}
        </text>
      </svg>
      <div className={`mt-0.5 text-xs font-medium tracking-wide ${band.text}`}>{band.name} risk</div>
      <div className="mt-2 grid grid-cols-4 gap-1 w-full max-w-[170px] text-[9px] text-slate-500 font-mono">
        <div className="text-center border-t-2 border-emerald-500 pt-1">0–25</div>
        <div className="text-center border-t-2 border-amber-500 pt-1">26–50</div>
        <div className="text-center border-t-2 border-orange-500 pt-1">51–75</div>
        <div className="text-center border-t-2 border-red-500 pt-1">76–100</div>
      </div>
    </div>
  );
}

function Meter({ label, value, max, unit, warn }) {
  const pct = Math.min(100, (value / max) * 100);
  const hot = pct > warn;
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex justify-between text-xs text-slate-400 mb-1">
        <span>{label}</span>
        <span className={`font-mono ${hot ? "text-red-400" : "text-slate-300"}`}>
          {value.toFixed(1)}
          {unit}
        </span>
      </div>
      <div className="h-1.5 bg-slate-800 overflow-hidden">
        <div
          className={`h-full ${hot ? "bg-red-500" : "bg-cyan-500"}`}
          style={{ width: `${pct}%`, transition: "width 400ms ease-out" }}
        />
      </div>
    </div>
  );
}

function TranscriptLine({ line, isNew }) {
  const isCounselor = line.speaker === "counselor";
  return (
    <div className={`px-3 py-2 border-b border-slate-800/60 ${isNew ? "bg-slate-800/40" : ""}`} style={{ transition: "background-color 800ms ease-out" }}>
      <div className="flex items-center gap-2 mb-0.5">
        <span className={`text-[11px] font-medium ${isCounselor ? "text-cyan-400" : "text-slate-300"}`}>
          {isCounselor ? "Counselor" : "Caller"}
        </span>
        {line.flag && (
          <span className="text-[10px] font-mono px-1.5 py-0.5 bg-red-500/10 text-red-400 border border-red-500/30">
            {line.flag}
          </span>
        )}
      </div>
      <p className="text-sm text-slate-200 leading-snug font-mono">{line.text}</p>
    </div>
  );
}

function ScriptRecommender({ band }) {
  const suggestions = {
    Low: [
      "Reflect back what she just told you before moving on.",
      "Confirm she's somewhere she can keep talking safely.",
    ],
    Moderate: [
      "Name the legal protection available under the SC/ST Act — don't wait for her to ask.",
      "Avoid asking her to repeat graphic detail; confirm what you already have instead.",
    ],
    High: [
      "Shift to short, closed questions — yes/no is easier under stress.",
      "Prepare her for a possible pause: 'If it gets unsafe to talk, tap the phone instead.'",
    ],
    Critical: [
      "Do not disconnect, even on silence — stay on an open line.",
      "Hand off location to dispatch now; keep her on the line rather than transferring the call.",
    ],
  };
  return (
    <div className="space-y-2">
      {suggestions[band.name].map((s, i) => (
        <div key={i} className="text-sm text-slate-300 border-l-2 border-cyan-500/50 pl-3 py-1">
          {s}
        </div>
      ))}
    </div>
  );
}

function ActionCards({ band, coords }) {
  const critical = band.name === "Critical";
  return (
    <div className="space-y-3">
      {!critical && (
        <div className="text-sm text-slate-500 border border-slate-800 p-4 text-center">
          No emergency dispatch required at current risk level.
        </div>
      )}
      {critical && (
        <>
          <div className="border border-red-500/40 bg-red-500/5">
            <div className="flex items-center gap-2 px-3 py-2 border-b border-red-500/30">
              <Siren size={14} className="text-red-400" />
              <span className="text-sm font-medium text-red-300">Police Dispatch — 112</span>
            </div>
            <div className="px-3 py-2 text-xs text-slate-300 space-y-1">
              <div className="flex items-center gap-1.5 font-mono text-slate-400">
                <MapPin size={12} /> {coords}
              </div>
              <div>Pre-filled from live GPS · awaiting supervisor ratification</div>
            </div>
            <button
              className="w-full text-xs font-medium py-2 bg-red-500/10 text-red-300 border-t border-red-500/30 cursor-not-allowed"
              title="Requires supervisor ratification (HITL gate)"
              disabled
            >
              Locked — supervisor sign-off required
            </button>
          </div>
          <div className="border border-orange-500/40 bg-orange-500/5">
            <div className="flex items-center gap-2 px-3 py-2 border-b border-orange-500/30">
              <ShieldAlert size={14} className="text-orange-400" />
              <span className="text-sm font-medium text-orange-300">Witness Protection Docket</span>
            </div>
            <div className="px-3 py-2 text-xs text-slate-300 space-y-1">
              <div>Section 398 BNSS · Bangalore Standing Committee</div>
              <div>5-working-day statutory response window</div>
            </div>
            <button
              className="w-full text-xs font-medium py-2 bg-orange-500/10 text-orange-300 border-t border-orange-500/30 cursor-not-allowed"
              title="Requires supervisor ratification (HITL gate)"
              disabled
            >
              Locked — supervisor sign-off required
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function AuditHistory() {
  return (
    <div className="border border-slate-800 bg-slate-900/40">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-800 bg-slate-900">
        <Fingerprint size={14} className="text-slate-400" />
        <span className="text-sm text-slate-300">Cryptographic audit trail</span>
      </div>
      <div className="divide-y divide-slate-800/60">
        {AUDIT_LOG.map((e, i) => (
          <div key={i} className="flex items-start gap-4 px-4 py-3 text-sm">
            <span className="font-mono text-slate-500 w-12 shrink-0">{e.time}</span>
            <span
              className={`w-20 shrink-0 text-xs font-medium ${
                e.actor === "Supervisor" ? "text-cyan-400" : e.actor === "AI" ? "text-slate-400" : "text-slate-600"
              }`}
            >
              {e.actor}
            </span>
            <span className="flex-1 text-slate-300">{e.action}</span>
            <span className="font-mono text-[11px] text-slate-600 shrink-0">#{e.hash}</span>
          </div>
        ))}
      </div>
      <div className="px-4 py-2 text-[11px] text-slate-600 border-t border-slate-800">
        Ledger is append-only and hash-signed — entries cannot be edited or deleted, per Bharatiya Sakshya Adhiniyam, 2023 evidentiary requirements.
      </div>
    </div>
  );
}

function AudioUploadBlock({ onAnalyzeComplete }) {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [status, setStatus] = useState(null);
  const fileInputRef = useRef(null);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      setAudioUrl(URL.createObjectURL(f));
      setStatus(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setStatus("Analyzing audio with faster-whisper & openSMILE...");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("http://127.0.0.1:8000/calls/live-call-01/analyze-audio", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setStatus(`Analyzed: SVI ${data.svi_metrics?.score || data.svi} (${data.category || data.band})`);
      if (onAnalyzeComplete) onAnalyzeComplete(data, file.name);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="border border-cyan-500/30 bg-slate-900/60 p-4 rounded mb-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Upload size={14} className="text-cyan-400" />
          <span className="text-xs font-semibold text-slate-200">Upload Call Audio / Evidence File</span>
        </div>
        <span className="text-[10px] text-slate-500 font-mono">WAV, MP3, FLAC</span>
      </div>
      <div className="flex flex-col sm:flex-row gap-3 items-center">
        <input
          type="file"
          accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a"
          ref={fileInputRef}
          onChange={handleFile}
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current && fileInputRef.current.click()}
          className="w-full sm:w-auto px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded"
        >
          {file ? file.name : "Select Audio File"}
        </button>
        {audioUrl && (
          <audio controls src={audioUrl} className="h-7 max-w-[180px] flex-1" />
        )}
        <button
          onClick={handleAnalyze}
          disabled={!file || isAnalyzing}
          className={`w-full sm:w-auto px-4 py-1.5 text-xs font-semibold rounded uppercase tracking-wider ${
            !file || isAnalyzing
              ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
              : "bg-cyan-600 hover:bg-cyan-500 text-white cursor-pointer shadow"
          }`}
        >
          {isAnalyzing ? "Analyzing..." : "Run GPU Triage"}
        </button>
      </div>
      {status && (
        <div className="mt-2 text-[11px] font-mono p-1.5 rounded bg-cyan-950/40 text-cyan-300 border border-cyan-500/20">
          {status}
        </div>
      )}
    </div>
  );
}

const TABS = [
  { id: "overview", label: "Overview", icon: LayoutGrid },
  { id: "upload", label: "Audio Ingestion", icon: Upload },
  { id: "transcript", label: "Transcript", icon: MessageSquareText },
  { id: "diagnostics", label: "Diagnostics", icon: Activity },
  { id: "guidance", label: "Guidance", icon: ShieldAlert },
  { id: "history", label: "History", icon: History },
];

export default function CounselorDashboard() {
  const [stepIndex, setStepIndex] = useState(0);
  const [lines, setLines] = useState([CALL_SCRIPT[0]]);
  const [activeTab, setActiveTab] = useState("overview");
  const elapsed = useElapsed();
  const scrollRef = useRef(null);

  useEffect(() => {
    const id = setInterval(() => {
      setStepIndex((i) => {
        const next = (i + 1) % CALL_SCRIPT.length;
        setLines((prev) => {
          const updated = next === 0 ? [CALL_SCRIPT[0]] : [...prev, CALL_SCRIPT[next]];
          return updated.slice(-8);
        });
        return next;
      });
    }, 3000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [lines]);

  const current = CALL_SCRIPT[stepIndex];
  const band = bandFor(current.svi);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      {/* Top bar */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800 bg-slate-900/60">
        <div className="flex items-center gap-3">
          <Phone size={16} className="text-cyan-400" />
          <span className="text-sm font-medium">Call #NHAA-2291</span>
          <span className="text-xs text-slate-500 font-mono">{elapsed} elapsed</span>
        </div>
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <div className="flex items-center gap-1.5">
            <Lock size={12} className="text-emerald-400" />
            RAM isolation active
          </div>
          <div className="flex items-center gap-1.5">
            <Radio size={12} className="text-cyan-400" />
            Dialect: Kannada / Kanglish
          </div>
          <div className={`px-2 py-0.5 border ${band.border} ${band.text} font-medium`}>{band.name} risk</div>
        </div>
      </div>

      {/* Tab nav */}
      <div className="flex items-center gap-1 px-4 pt-3 border-b border-slate-800 bg-slate-950">
        {TABS.map((t) => {
          const Icon = t.icon;
          const active = activeTab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`flex items-center gap-1.5 px-3 py-2 text-sm border-b-2 -mb-px ${
                active
                  ? "border-cyan-500 text-slate-100"
                  : "border-transparent text-slate-500 hover:text-slate-300"
              }`}
            >
              <Icon size={13} />
              {t.label}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div className="p-4">
        {activeTab === "overview" && (
          <div>
            <AudioUploadBlock onAnalyzeComplete={(data) => {
              if (data.nlp_telemetry?.transcript) {
                setLines(prev => [...prev, { speaker: "caller", text: data.nlp_telemetry.transcript, flag: data.category === "CRITICAL_RISK" ? "critical override" : null }]);
              }
            }} />
            <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_1fr_1fr] gap-4">
            <div className="border border-slate-800 bg-slate-900/40 flex flex-col h-[560px]">
              <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-800 bg-slate-900">
                <MessageSquareText size={14} className="text-slate-400" />
                <span className="text-sm text-slate-300">Live transcription</span>
                <span className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              </div>
              <div ref={scrollRef} className="flex-1 overflow-y-auto">
                {lines.map((l, i) => (
                  <TranscriptLine key={i} line={l} isNew={i === lines.length - 1} />
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-4">
              <div className="border border-slate-800 bg-slate-900/40 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Gauge size={14} className="text-slate-400" />
                  <span className="text-sm text-slate-300">Stress Vulnerability Index</span>
                </div>
                <SVIGauge svi={current.svi} />
              </div>
              <div className="border border-slate-800 bg-slate-900/40 p-4 flex-1">
                <div className="flex items-center gap-2 mb-3">
                  <Activity size={14} className="text-slate-400" />
                  <span className="text-sm text-slate-300">Acoustic diagnostics</span>
                </div>
                <Meter label="Jitter (voice cracking)" value={current.jitter} max={4} unit="%" warn={70} />
                <Meter label="Shimmer (sobbing)" value={current.shimmer} max={8} unit="%" warn={70} />
                <Meter label="Harmonics-to-noise ratio" value={current.hnr} max={24} unit=" dB" warn={200} />
                <Meter label="F0 pitch variance" value={current.f0} max={40} unit=" Hz" warn={70} />
              </div>
            </div>

            <div className="flex flex-col gap-4">
              <div className="border border-slate-800 bg-slate-900/40 p-4">
                <div className="flex items-center gap-2 mb-3">
                  <MessageSquareText size={14} className="text-slate-400" />
                  <span className="text-sm text-slate-300">Suggested approach</span>
                </div>
                <ScriptRecommender band={band} />
              </div>
              <div className="border border-slate-800 bg-slate-900/40 p-4 flex-1">
                <div className="flex items-center gap-2 mb-3">
                  <ShieldAlert size={14} className="text-slate-400" />
                  <span className="text-sm text-slate-300">Emergency actions</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "upload" && (
          <div className="max-w-2xl mx-auto p-4">
            <AudioUploadBlock onAnalyzeComplete={(data) => {
              if (data.nlp_telemetry?.transcript) {
                setLines(prev => [...prev, { speaker: "caller", text: data.nlp_telemetry.transcript, flag: data.category === "CRITICAL_RISK" ? "critical override" : null }]);
              }
            }} />
          </div>
        )}

        {activeTab === "transcript" && (
          <div className="border border-slate-800 bg-slate-900/40 flex flex-col h-[640px]">
            <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-800 bg-slate-900">
              <MessageSquareText size={14} className="text-slate-400" />
              <span className="text-sm text-slate-300">Live transcription</span>
              <span className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            </div>
            <div ref={scrollRef} className="flex-1 overflow-y-auto">
              {lines.map((l, i) => (
                <TranscriptLine key={i} line={l} isNew={i === lines.length - 1} />
              ))}
            </div>
          </div>
        )}

        {activeTab === "diagnostics" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border border-slate-800 bg-slate-900/40 p-6 flex justify-center">
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Gauge size={14} className="text-slate-400" />
                  <span className="text-sm text-slate-300">Stress Vulnerability Index</span>
                </div>
                <SVIGauge svi={current.svi} />
              </div>
            </div>
            <div className="border border-slate-800 bg-slate-900/40 p-4">
              <div className="flex items-center gap-2 mb-3">
                <Activity size={14} className="text-slate-400" />
                <span className="text-sm text-slate-300">Acoustic diagnostics</span>
              </div>
              <Meter label="Jitter (voice cracking)" value={current.jitter} max={4} unit="%" warn={70} />
              <Meter label="Shimmer (sobbing)" value={current.shimmer} max={8} unit="%" warn={70} />
              <Meter label="Harmonics-to-noise ratio" value={current.hnr} max={24} unit=" dB" warn={200} />
              <Meter label="F0 pitch variance" value={current.f0} max={40} unit=" Hz" warn={70} />
            </div>
          </div>
        )}

        {activeTab === "guidance" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border border-slate-800 bg-slate-900/40 p-4">
              <div className="flex items-center gap-2 mb-3">
                <MessageSquareText size={14} className="text-slate-400" />
                <span className="text-sm text-slate-300">Suggested approach</span>
              </div>
              <ScriptRecommender band={band} />
            </div>
            <div className="border border-slate-800 bg-slate-900/40 p-4">
              <div className="flex items-center gap-2 mb-3">
                <ShieldAlert size={14} className="text-slate-400" />
                <span className="text-sm text-slate-300">Emergency actions</span>
              </div>
              <ActionCards band={band} coords="12.9716° N, 77.5946° E" />
            </div>
          </div>
        )}

        {activeTab === "history" && <AuditHistory />}
      </div>

      <div className="px-5 pb-4 text-[11px] text-slate-600">
        Simulated data for demo purposes — no live audio pipeline connected. All dispatch actions are advisory and require human ratification (HITL).
      </div>
    </div>
  );
}
