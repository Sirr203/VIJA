// Hands-free mode: listens continuously, segments speech by silence, then invokes onUtterance.
// Core logic (Segmenter, resample) has no browser dependencies, making it unit-testable via Node.js.

class Segmenter {
  constructor(o = {}) {
    this.threshold = o.threshold ?? 0.015;      // RMS threshold to consider as speech
    this.endSilenceMs = o.endSilenceMs ?? 800;  // Duration of silence required to commit utterance
    this.minSpeechMs = o.minSpeechMs ?? 250;    // Shorter durations are considered noise
    this.maxMs = o.maxMs ?? 15000;              // Force cut if speech exceeds this duration
    this.prerollFrames = o.prerollFrames ?? 8;  // Retain ~0.35s of audio prior to speech onset
    this.level = 0;
    this.reset();
  }

  reset() {
    this.preroll = [];
    this.frames = [];
    this.speaking = false;
    this.silenceMs = 0;
    this.speechMs = 0;
    this.totalMs = 0;
  }

  // Accepts a Float32Array audio frame; returns Float32Array of complete utterance when committed, else null.
  push(frame, sampleRate) {
    const ms = (frame.length / sampleRate) * 1000;
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    this.level = Math.sqrt(sum / frame.length);
    const loud = this.level > this.threshold;

    if (!this.speaking) {
      this.preroll.push(frame);
      if (this.preroll.length > this.prerollFrames) this.preroll.shift();
      if (!loud) return null;
      this.speaking = true;
      this.frames = this.preroll;
      this.preroll = [];
      this.totalMs = this.frames.length * ms;
      this.speechMs = ms;
      this.silenceMs = 0;
      return null;
    }

    this.frames.push(frame);
    this.totalMs += ms;
    if (loud) { this.speechMs += ms; this.silenceMs = 0; } else { this.silenceMs += ms; }
    if (this.silenceMs < this.endSilenceMs && this.totalMs < this.maxMs) return null;

    const out = this.speechMs >= this.minSpeechMs ? concat(this.frames) : null;
    this.reset();
    return out;
  }
}

function concat(frames) {
  let n = 0;
  for (const f of frames) n += f.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const f of frames) { out.set(f, o); o += f.length; }
  return out;
}

// Downsample audio by averaging chunk blocks (effective for voice audio)
function resample(input, from, to) {
  if (from === to) return input;
  const ratio = from / to;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const a = Math.floor(i * ratio);
    const b = Math.min(input.length, Math.floor((i + 1) * ratio));
    let s = 0;
    for (let j = a; j < b; j++) s += input[j];
    out[i] = s / Math.max(1, b - a);
  }
  return out;
}

const TAP_PROCESSOR = `
class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
  process(inputs) {
    const ch = inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.buf[this.n++] = ch[i];
        if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice(0)); this.n = 0; }
      }
    }
    return true;
  }
}
registerProcessor("tap", Tap);
`;

// Activate mic and listen continuously. Returns { stop(), pause(), resume() }. Throws if mic permission fails.
async function startHandsFree({ onUtterance, onLevel, getThreshold }) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true },
  });
  const ctx = new AudioContext();
  const seg = new Segmenter();
  try {
    const url = URL.createObjectURL(new Blob([TAP_PROCESSOR], { type: "text/javascript" }));
    try { await ctx.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
    await ctx.resume();

    const source = ctx.createMediaStreamSource(stream);
    const tap = new AudioWorkletNode(ctx, "tap");
    const mute = ctx.createGain();   // Connect to destination with 0 gain to ensure browser runs worklet
    mute.gain.value = 0;
    source.connect(tap);
    tap.connect(mute).connect(ctx.destination);

    tap.port.onmessage = (e) => {
      if (getThreshold) seg.threshold = getThreshold();
      const utterance = seg.push(e.data, ctx.sampleRate);
      if (onLevel) onLevel(seg.level, seg.speaking);
      if (utterance) onUtterance(utterance, ctx.sampleRate);
    };
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    ctx.close();
    throw err;
  }
  return {
    stop() {
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
    },
    pause() {
      seg.reset();
      if (ctx.state === "running") ctx.suspend();
    },
    resume() {
      seg.reset();
      if (ctx.state === "suspended") ctx.resume();
    },
  };
}

if (typeof module !== "undefined") module.exports = { Segmenter, resample, startHandsFree };
