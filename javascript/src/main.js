import * as THREE from "three";

// Create a scene
const scene = new THREE.Scene();

// Create a camera
const camera = new THREE.PerspectiveCamera(
  60, // Field of view
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
const redMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
const greenMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
const blueMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff });

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
greenSphere.position.copy(bottomLeftPosition);
const blueSphere = new THREE.Mesh(sphereGeometry, blueMaterial);
blueSphere.position.copy(bottomRightPosition);

// Add spheres to the scene
scene.add(redSphere);
scene.add(greenSphere);
scene.add(blueSphere);

// Position the camera
camera.position.set(0, centerHeight, 18);
camera.lookAt(0, centerHeight, 0);

// Initialize CCapture for video recording
const capturer = new CCapture({ format: "webm", framerate: 60 });
let isCapturing = false;

// Start capturing when the animation begins
function startCapture() {
  if (!isCapturing) {
    capturer.start();
    isCapturing = true;
  }
}

// Stop capturing and save the video
function stopCapture() {
  if (isCapturing) {
    capturer.stop();
    capturer.save();
    isCapturing = false;
  }
}

// Animation loop
let cameraAngle = -Math.PI; // Start at (0, 18, 18)
function animate() {
  if (cameraAngle < Math.PI) {
    requestAnimationFrame(animate);

    // Start capturing
    startCapture();

    // Move the camera in a circle in the x, z plane
    cameraAngle += 0.01;
    const cameraRadius = 8;
    const circleCenter = { x: 0, y: 18, z: 18 + 8 };
    camera.position.x = circleCenter.x + Math.sin(cameraAngle) * cameraRadius;
    camera.position.y = circleCenter.y;
    camera.position.z = circleCenter.z + Math.cos(cameraAngle) * cameraRadius;
    // camera.lookAt(0, centerHeight, 0);
    let offset = 0;
    if (cameraAngle > -Math.PI / 2) {
      offset = 3;
    }
    if (cameraAngle > 0) {
      offset = -3;
    }
    if (cameraAngle > Math.PI / 2) {
      offset = 0;
    }

    camera.lookAt(offset, centerHeight, 0);

    renderer.render(scene, camera);

    // Capture the current frame
    capturer.capture(renderer.domElement);
  } else {
    // Stop capturing when the animation ends
    stopCapture();
  }
}

animate();
