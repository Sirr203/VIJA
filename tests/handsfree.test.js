// Run with: node --test tests/
const test = require("node:test");
const assert = require("node:assert");
const { Segmenter, resample } = require("../app/static/handsfree.js");

const RATE = 48000, N = 2048;                 // each frame ≈ 42.7 ms
const quiet = () => new Float32Array(N).fill(0.001);
const loud = () => new Float32Array(N).fill(0.2);
const run = (seg, frames) => frames.map((f) => seg.push(f, RATE)).filter(Boolean);

test("silence only yields no utterance and preroll remains bounded", () => {
  const seg = new Segmenter();
  assert.deepStrictEqual(run(seg, Array.from({ length: 100 }, quiet)), []);
  assert.ok(seg.preroll.length <= seg.prerollFrames);
});

test("speech followed by sufficient silence commits an utterance with preroll", () => {
  const seg = new Segmenter();
  const frames = [...Array(10).fill(0).map(quiet), ...Array(30).fill(0).map(loud), ...Array(25).fill(0).map(quiet)];
  const out = run(seg, frames);
  assert.strictEqual(out.length, 1);
  assert.ok(out[0].length >= 30 * N + 8 * N);   // 30 loud frames + 8 preroll frames
  assert.strictEqual(seg.speaking, false);      // reset and ready for next utterance
});

test("short noise below minSpeechMs is discarded", () => {
  const seg = new Segmenter();
  const frames = [loud(), ...Array(30).fill(0).map(quiet)];
  assert.deepStrictEqual(run(seg, frames), []);
});

test("brief pause within a sentence does not split the utterance", () => {
  const seg = new Segmenter();
  const frames = [...Array(15).fill(0).map(loud), ...Array(8).fill(0).map(quiet), ...Array(15).fill(0).map(loud), ...Array(25).fill(0).map(quiet)];
  assert.strictEqual(run(seg, frames).length, 1);   // 8 frames ≈ 340 ms < 800 ms
});

test("speech exceeding maxMs is forcefully cut", () => {
  const seg = new Segmenter({ maxMs: 1000 });
  const out = run(seg, Array.from({ length: 40 }, loud));
  assert.ok(out.length >= 1);
});

test("resample 48k -> 16k preserves duration and values", () => {
  const out = resample(new Float32Array(48000).fill(0.5), 48000, 16000);
  assert.strictEqual(out.length, 16000);
  assert.ok(Math.abs(out[100] - 0.5) < 1e-6);
  assert.strictEqual(resample(new Float32Array(10), 16000, 16000).length, 10);
});
