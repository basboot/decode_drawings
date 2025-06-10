import * as THREE from "three";

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

export class Disturbance {
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

export class RandomDisturbance {
  x = 0;
  y = 0;
  rot = 0;

  maxX;
  maxY;
  maxRotation;

  constructor(maxX, maxY, maxRotation) {
    this.maxX = maxX; // Maximum disturbance in the X direction
    this.maxY = maxY; // Maximum disturbance in the Y direction
    this.maxRotation = maxRotation; // Maximum rotational disturbance

    this.update();
  }

  update() {
    this.x = THREE.MathUtils.randFloat(-this.maxX, this.maxX);
    this.y = THREE.MathUtils.randFloat(-this.maxY, this.maxY);
    this.rot = THREE.MathUtils.randFloat(-this.maxRotation, this.maxRotation);
  }

  disturbCamera(camera) {
    // Apply disturbances to the camera using x, y, and rot
    disturb_camera(camera, this.x, this.y, this.rot);
  }
}
