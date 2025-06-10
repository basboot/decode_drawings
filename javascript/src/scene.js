import * as THREE from "three";
import {centerHeight, radius, triangleHeight, triangleSide} from "./config.js";

export function setupScene() {
    // Create a container with fixed dimensions
const container = document.createElement("div");
container.style.position = "relative";
container.style.width = "1280px";
container.style.height = "720px";
container.style.margin = "0 auto"; // Center horizontally
document.body.appendChild(container);

// Create a scene
const scene = new THREE.Scene();

// Create a camera
const camera = new THREE.PerspectiveCamera(
  60, // VERTICAL fov, 60 degrees gives the best (visual) match
  1280 / 720, // Aspect ratio
  0.1, // Near clipping plane
  100 // Far clipping plane
);

// Create a renderer with fixed size
const renderer = new THREE.WebGLRenderer();
renderer.setSize(1280, 720);
renderer.domElement.style.width = "1280px";
renderer.domElement.style.height = "720px";
renderer.domElement.style.display = "block"; // Ensure block display

// Append the renderer to the container instead of directly to body
container.appendChild(renderer.domElement);

return [container, scene, camera, renderer];
}

export // Add background image overlay with 50% transparency
function addImageOverlay(imageSrc, container) {
  console.log("Adding overlay with image:", imageSrc);

  // Create a container for the overlay
  const overlayContainer = document.createElement("div");
  overlayContainer.style.position = "absolute";
  overlayContainer.style.top = "0";
  overlayContainer.style.left = "0";
  overlayContainer.style.width = "1280px"; // Fixed width
  overlayContainer.style.height = "720px"; // Fixed height
  overlayContainer.style.pointerEvents = "none"; // Allow click-through
  overlayContainer.style.zIndex = "10"; // Ensure it's above the canvas

  // Create the image element
  const overlayImage = document.createElement("img");
  overlayImage.src = imageSrc;
  overlayImage.style.width = "1280px"; // Fixed width
  overlayImage.style.height = "720px"; // Fixed height
  overlayImage.style.objectFit = "cover";
  overlayImage.style.opacity = "0.5"; // 50% transparency

  // Add error handling to check if the image loads
  overlayImage.onload = function () {
    console.log("Image loaded successfully!");
  };

  overlayImage.onerror = function () {
    console.error("Failed to load image:", imageSrc);
    // Try with an absolute path
    console.log("Attempting with absolute path...");
    const baseUrl = window.location.href.substring(0, window.location.href.lastIndexOf("/") + 1);
    overlayImage.src = baseUrl + imageSrc;
  };

  // Overlay to check ball sizes to tune fov
  // overlayContainer.appendChild(overlayImage);

  // Add overlay to the main container (not body)
  container.appendChild(overlayContainer);

  return overlayContainer;
}

export function addTriangleToScene(scene) {
    // Add three spheres in an equilateral triangle

const redMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 }); // Basic material without lighting
const greenMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 }); // Basic material without lighting
const blueMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff }); // Basic material without lighting

// Calculate positions for the equilateral triangle
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
}

export // Log the current position and viewing angle of the camera
function logCameraInfo(camera) {
  console.log("Camera Position:", camera.position);
  console.log("Camera Rotation (in radians):", camera.rotation);
  console.log("Camera Rotation (in degrees):");
}