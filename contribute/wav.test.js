// Run: node contribute/wav.test.js   (writes a WAV to the path in argv[2] for a Python check)
const assert = require("assert");
const fs = require("fs");
const W = require("./wav.js");

const tone = (sec, rate, amp = 0.5) =>
  Float32Array.from({ length: sec * rate }, (_, i) => amp * Math.sin((2 * Math.PI * 440 * i) / rate));

const down = W.resample(tone(2, 48000), 48000);
assert.strictEqual(down.length, 32000);
assert.strictEqual(W.check(down), null);
assert.strictEqual(W.check(new Float32Array(8000)), "short");
assert.strictEqual(W.check(new Float32Array(32000)), "quiet");
assert.strictEqual(W.check(tone(2, 16000, 1.5).map((v) => Math.max(-1, Math.min(1, v)))), "clipped");

const wav = Buffer.from(W.encode(down));
assert.strictEqual(wav.length, 44 + 64000);
assert.strictEqual(wav.toString("ascii", 0, 4), "RIFF");
if (process.argv[2]) fs.writeFileSync(process.argv[2], wav);
console.log("wav.js ok");
