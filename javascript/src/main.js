// https://community.gopro.com/s/article/HERO8-Black-Digital-Lenses-formerly-known-as-FOV?language=en_US

// maxima in drawings
// td rom 16 to 35 -> center 26 +- 11
// lr -10.5, 10.5

import * as THREE from "three";
import { addImageOverlay, addTriangleToScene, logCameraInfo, setupScene } from "./scene.js";
import { centerHeight } from "./config.js";
import { Disturbance, RandomDisturbance } from "./disturbance.js";
import { Capturer } from "./capture.js";
import { circle, initialPosition, sweepPaper } from "./animations.js";

const [container, scene, camera, renderer] = setupScene();

// const overlay = addImageOverlay("/video_1_frame_0000.png", container);

addTriangleToScene(scene);

// Position the camera
camera.position.set(0, centerHeight, 18);
camera.lookAt(0, centerHeight, 0);

// x 1, 0.01
// rot 2, 0.03
// const disturbance = new Disturbance(1.5, 1.5, 2.5, 0.1, 0.1, 0.05);

const randomDisturbance = new RandomDisturbance(10, 10, 10);

// create a link to the real Date.now, because the capturer will overwrite
// it to fake realtime during capturing
const dateNow = Date.now;

const startTime = dateNow(); // store ts of start

const capturer = new Capturer();
capturer.DEBUG = false;

// const cameraPositions = sweepPaper(0.1);

// const cameraPositions = initialPosition();

const cameraPositions = circle();

let animationPosition = 0;
const cameraBasePositions = [];

console.log(`Starting simulation to create ${cameraPositions.length} datapoints.`);

// Animation loop
function animate() {
  if (animationPosition < cameraPositions.length) {
    capturer.startCapture();

    // disturbance.update();
    randomDisturbance.update();

    // update camera position
    camera.position.x = cameraPositions[animationPosition].x;
    camera.position.y = cameraPositions[animationPosition].y;
    camera.position.z = cameraPositions[animationPosition].z;
    // camera.position.x = 0;
    // camera.position.y = 18;
    // camera.position.z = 18;

    // update cameraAngle to look at triangle
    camera.lookAt(0, centerHeight, 0);

    // store camera base, right below the camera
    cameraBasePositions.push([camera.position.x, 0, camera.position.z]);

    // add disturbance
    // disturbance.disturbCamera(camera);
    // randomDisturbance.disturbCamera(camera);

    renderer.render(scene, camera);

    // logCameraInfo(camera);

    capturer.captureFrame(renderer);

    // log progress every 600 frames (10 animation seconds, not real seconds)
    if (animationPosition % 600 === 0 && animationPosition > 0) {
      const runningTime = dateNow() - startTime;
      const progress = animationPosition / cameraPositions.length;
      const estimatedTime = runningTime / progress - runningTime;
      console.log(`${(100 * progress).toFixed(2)} % done`);
      const remainingMinutes = Math.floor(estimatedTime / 1000 / 60);
      const remainingSeconds = Math.floor((estimatedTime / 1000) % 60);
      console.log(`Estimated time remaining: ${remainingMinutes} minutes and ${remainingSeconds} seconds`);
    }

    animationPosition++;

    requestAnimationFrame(animate);
  } else {
    // Stop capturing when the animation ends
    console.log("Ready");
    capturer.stopCapture(cameraBasePositions);
  }
}

animate();
