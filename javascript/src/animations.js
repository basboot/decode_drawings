import { centerHeight, paperCenter, paperSize } from "./config.js";

export function sweepPaper(stepSize = 0.05) {
  let cameraPositions = [];

  for (let z = paperCenter - paperSize; z < paperCenter + paperSize; z += stepSize) {
    for (let x = -paperSize; x < paperSize; x += stepSize) {
      cameraPositions.push({ x: x, y: centerHeight, z: z });
    }
  }

  return cameraPositions;
}

export function initialPosition(n = 120) {
  let cameraPositions = [];

  for (let i = 0; i < n; i++) {
    cameraPositions.push({ x: 0, y: centerHeight, z: 18 });
  }

  return cameraPositions;
}

export function circle(r = 8, o = 18, frames = 2301) {
  let cameraPositions = [];

  for (let i = 0; i < frames; i++) {
    let angle = (2 * Math.PI * i) / frames - Math.PI;
    let x = r * Math.sin(-angle);
    let y = r * Math.cos(-angle) + o + r;
    cameraPositions.push({ x: x, y: centerHeight, z: y });
  }

  return cameraPositions;
}
