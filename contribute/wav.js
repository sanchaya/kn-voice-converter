// Audio helpers for the contribute page: downsample to 16 kHz, quality checks, 16-bit mono WAV.
(function (root) {
  const RATE = 16000;

  // Box-filter decimation; fine for 48k/44.1k mic input.
  function resample(f32, from) {
    if (from === RATE) return f32;
    const r = from / RATE, out = new Float32Array(Math.floor(f32.length / r));
    for (let i = 0; i < out.length; i++) {
      const a = Math.floor(i * r), b = Math.max(a + 1, Math.floor((i + 1) * r));
      let s = 0;
      for (let j = a; j < b; j++) s += f32[j];
      out[i] = s / (b - a);
    }
    return out;
  }

  // Returns null if the clip is usable, else a reason key.
  function check(f32) {
    const seconds = f32.length / RATE;
    if (seconds < 1) return "short";
    if (seconds > 60) return "long";
    let peak = 0, clipped = 0;
    for (const v of f32) {
      const a = Math.abs(v);
      if (a > peak) peak = a;
      if (a >= 0.999) clipped++;
    }
    if (peak < 0.02) return "quiet";
    if (clipped / f32.length > 0.01) return "clipped";
    return null;
  }

  function encode(f32) {
    const buf = new ArrayBuffer(44 + f32.length * 2), v = new DataView(buf);
    const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    str(0, "RIFF"); v.setUint32(4, 36 + f32.length * 2, true); str(8, "WAVEfmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, RATE, true); v.setUint32(28, RATE * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, f32.length * 2, true);
    f32.forEach((s, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
    return buf;
  }

  root.KNWav = { RATE, resample, check, encode };
  if (typeof module !== "undefined") module.exports = root.KNWav;
})(typeof self !== "undefined" ? self : globalThis);
