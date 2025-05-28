// https://community.gopro.com/s/article/HERO8-Black-Digital-Lenses-formerly-known-as-FOV?language=en_US

// maxima in drawings
// td rom 16 to 35 -> center 26 +- 11
// lr -10.5, 10.5

import * as THREE from "three";

// Create a scene
const scene = new THREE.Scene();

// Create a camera
const camera = new THREE.PerspectiveCamera(
  59, // Field of view, needs more tuning maybe
  1280 / 720, // Aspect ratio
  0.1, // Near clipping plane
  1000 // Far clipping plane
);

// Create a renderer
const renderer = new THREE.WebGLRenderer();
renderer.setSize(1280, 720);
document.body.appendChild(renderer.domElement);

// Remove the cube and add three spheres in an equilateral triangle
const radius = 3;
const redMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 }); // Basic material without lighting
const greenMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 }); // Basic material without lighting
const blueMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff }); // Basic material without lighting

// Calculate positions for the equilateral triangle
const centerHeight = 18;
const triangleSide = 9;
const triangleHeight = (triangleSide * Math.sqrt(3)) / 2;
const topPosition = new THREE.Vector3(0, centerHeight + (2 / 3) * triangleHeight, 0);
const bottomLeftPosition = new THREE.Vector3(-triangleSide / 2, centerHeight - (1 / 3) * triangleHeight, 0);
const bottomRightPosition = new THREE.Vector3(triangleSide / 2, centerHeight - (1 / 3) * triangleHeight, 0);

// Create the spheres
const sphereGeometry = new THREE.SphereGeometry(radius, 32, 32);
const redSphere = new THREE.Mesh(sphereGeometry, redMaterial);
redSphere.position.copy(topPosition);
const greenSphere = new THREE.Mesh(sphereGeometry, greenMaterial);
greenSphere.position.copy(bottomRightPosition);
const blueSphere = new THREE.Mesh(sphereGeometry, blueMaterial);
blueSphere.position.copy(bottomLeftPosition);

// Add spheres to the scene
scene.add(redSphere);
scene.add(greenSphere);
scene.add(blueSphere);

// Position the camera
camera.position.set(0, centerHeight, 18);
camera.lookAt(0, centerHeight, 0);

// Log the current position and viewing angle of the camera
function logCameraInfo() {
  console.log("Camera Position:", camera.position);
  console.log("Camera Rotation (in radians):", camera.rotation);
  console.log("Camera Rotation (in degrees):");
}

// logCameraInfo();

function disturb_camera(camera, x_disturbance, z_disturbance, yaw_disturbance) {
  // Rotate/turn the camera around the base of the pole

  // assume camera was standing straight before disturbing
  const cameraBase = new THREE.Vector3(camera.position.x, 0, camera.position.z);

  // first rotate the pole (yaw)
  camera.rotation.z += THREE.MathUtils.degToRad(yaw_disturbance);

  // then simulate pushing the top of the poll in z or x direction

  // Create rotation matrices
  const zRotation = new THREE.Matrix4().makeRotationZ(THREE.MathUtils.degToRad(z_disturbance));
  const xRotation = new THREE.Matrix4().makeRotationX(THREE.MathUtils.degToRad(x_disturbance));

  // Translate the pole and camera to the origin, apply rotations, and translate back
  const translationToOrigin = new THREE.Matrix4().makeTranslation(-cameraBase.x, -cameraBase.y, -cameraBase.z);
  const translationBack = new THREE.Matrix4().makeTranslation(cameraBase.x, cameraBase.y, cameraBase.z);

  const combinedTransform = new THREE.Matrix4()
    .multiply(translationToOrigin)
    .multiply(zRotation)
    .multiply(xRotation)
    .multiply(translationBack);

  // Apply the transformation to the pole and camera
  camera.applyMatrix4(combinedTransform);
}

class Disturbance {
  x = 0;
  y = 0;
  rot = 0;

  maxX;
  maxY;
  maxRotation;

  maxSpeedX;
  maxSpeedY;
  maxSpeedRotation;

  speedX = 0;
  speedY = 0;
  speedRotation = 0;

  targetX = 0;
  targetY = 0;
  targetRotation = 0;

  constructor(maxX, maxY, maxRotation, maxSpeedX, maxSpeedY, maxSpeedRotation) {
    this.maxX = maxX; // Maximum disturbance in the X direction
    this.maxY = maxY; // Maximum disturbance in the Y direction
    this.maxRotation = maxRotation; // Maximum rotational disturbance

    this.maxSpeedX = maxSpeedX; // Maximum speed of disturbance in the X direction
    this.maxSpeedY = maxSpeedY; // Maximum speed of disturbance in the Y direction
    this.maxSpeedRotation = maxSpeedRotation; // Maximum speed of rotational disturbance

    this.setTargets();
  }

  setTargets(setX = true, setY = true, setRotation = true) {
    // Set random targets within the range [-max, max] based on parameters
    if (setX) {
      this.targetX = THREE.MathUtils.randFloat(-this.maxX, this.maxX);
      this.speedX = THREE.MathUtils.randFloat(0, this.maxSpeedX) * Math.sign(this.targetX - this.x);
    }
    if (setY) {
      this.targetY = THREE.MathUtils.randFloat(-this.maxY, this.maxY);
      this.speedY = THREE.MathUtils.randFloat(0, this.maxSpeedY) * Math.sign(this.targetY - this.y);
    }
    if (setRotation) {
      this.targetRotation = THREE.MathUtils.randFloat(-this.maxRotation, this.maxRotation);
      this.speedRotation =
        THREE.MathUtils.randFloat(0, this.maxSpeedRotation) * Math.sign(this.targetRotation - this.rot);
    }
  }

  update() {
    // Update x position
    if (Math.abs(this.x - this.targetX) <= Math.abs(this.speedX)) {
      this.x = this.targetX; // Snap to target
      this.setTargets(true, false, false); // Set new target for x
    } else {
      this.x += this.speedX; // Move towards target
    }

    // Update y position
    if (Math.abs(this.y - this.targetY) <= Math.abs(this.speedY)) {
      this.y = this.targetY; // Snap to target
      this.setTargets(false, true, false); // Set new target for y
    } else {
      this.y += this.speedY; // Move towards target
    }

    // Update rotation
    if (Math.abs(this.rot - this.targetRotation) <= Math.abs(this.speedRotation)) {
      this.rot = this.targetRotation; // Snap to target
      this.setTargets(false, false, true); // Set new target for rotation
    } else {
      this.rot += this.speedRotation; // Move towards target
    }
  }

  disturbCamera(camera) {
    // Apply disturbances to the camera using x, y, and rot
    disturb_camera(camera, this.x, this.y, this.rot);
  }
}

// Call the function to log camera info
// logCameraInfo();

// Initialize CCapture for video recording
const capturer = new CCapture({ format: "webm", framerate: 60 });
let isCapturing = false;

const DEBUG = false;

// Start capturing when the animation begins
function startCapture() {
  if (!isCapturing && !DEBUG) {
    capturer.start();
    isCapturing = true;
  }
}

let cameraBasePositions = [];

// Stop capturing and save the video
function stopCapture() {
  if (isCapturing && !DEBUG) {
    capturer.stop();
    capturer.save();

    // Download cameraBasePositions as JSON
    const blob = new Blob([JSON.stringify(cameraBasePositions, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "cameraBasePositions.json";
    link.click();
    isCapturing = false;
  }
}

// Animation loop
let cameraAngle = -Math.PI; // Start at (0, 18, 18)

// x 1, 0.01
// rot 2, 0.03
const disturbance = new Disturbance(1.5, 1.5, 2.5, 0.1, 0.1, 0.05);

const paperSize = 11;
const paperCenter = 26.5;
let cameraPosition = { x: -paperSize, y: 18, z: paperCenter - paperSize };
const stepSize = 0.1;

console.log(`Starting simulation to create ${paperCenter * paperCenter * 4 * (1 / stepSize)} datapoints.`);

const startTime = Date.now(); // Set epoch time

function animate() {
  if (cameraPosition.z < paperCenter + paperSize) {
    // Start capturing
    startCapture();

    disturbance.update();

    // update camera position
    // TODO: variable cameraPosition is redundant
    camera.position.x = cameraPosition.x;
    camera.position.y = cameraPosition.y;
    camera.position.z = cameraPosition.z;
    // camera.position.x = 0;
    // camera.position.y = 18;
    // camera.position.z = 18;

    // update cameraAngle
    camera.lookAt(0, centerHeight, 0);

    // store camera base, right below the camera
    cameraBasePositions.push([camera.position.x, 0, camera.position.z]);

    // add disturbance
    disturbance.disturbCamera(camera);

    renderer.render(scene, camera);

    capturer.capture(renderer.domElement);

    cameraPosition.x += stepSize;
    if (cameraPosition.x > paperSize) {
      cameraPosition.x = -paperSize;
      cameraPosition.z += stepSize;
      const pct = (cameraPosition.z - paperCenter + paperSize) / (2 * paperSize);
      const runningTime = Date.now() - startTime;
      const estimatedTime = runningTime / pct - runningTime;
      console.log(`${(100 * pct).toFixed(2)} % done`);
      const remainingMinutes = Math.floor(estimatedTime / 1000 / 60);
      const remainingSeconds = Math.floor((estimatedTime / 1000) % 60);
      console.log(`Estimated time remaining: ${remainingMinutes} minutes and ${remainingSeconds} seconds`);
    }

    requestAnimationFrame(animate);
  } else {
    // Stop capturing when the animation ends
    console.log("Ready");
    stopCapture();
  }
}

animate();
