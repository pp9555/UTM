#include "HX711.h"

// --- Pin Definitions ---
const int LOADCELL_DOUT_PIN = 3;
const int LOADCELL_SCK_PIN = 2;
const int DIR_PIN = 4;
const int STEP_PIN = 5;

HX711 scale;
float calibration_factor = 420.0; 

// Motor & Test State Variables
unsigned long previousMotorTime = 0;
const long motorInterval = 5; 
bool motorState = LOW;

// --- Auto-Stop Logic Variables ---
bool isTesting = true;         
float maxForce = 0.0;          
float armingThreshold = 0.5;   // Force must exceed 0.5kg before auto-stop is allowed to trigger
float fractureThreshold = 0.6; // If force drops to 60% of the max force, trigger shutoff

void setup() {
  Serial.begin(115200);
  
  pinMode(DIR_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  digitalWrite(DIR_PIN, HIGH); 
  
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_scale(calibration_factor);
  scale.tare(); 

  Serial.println("System Ready. Starting Test...");
}

void loop() {
  unsigned long currentMillis = millis();

  // 1. MOTOR CONTROL (Only spins if the test is active)
  if (isTesting && (currentMillis - previousMotorTime >= motorInterval)) {
    previousMotorTime = currentMillis;
    motorState = !motorState;
    digitalWrite(STEP_PIN, motorState);
  }

  // 2. READ & ANALYZE FORCE
  if (scale.is_ready() && isTesting) {
    // Read an average of 3 samples to filter out electronic noise
    float currentForce = scale.get_units(3); 
    
    // Ignore tiny negative sensor drift at the start
    if (currentForce < 0) currentForce = 0; 
    
    Serial.print("FORCE:");
    Serial.println(currentForce);

    // Track the Peak Force (Highest value reached so far)
    if (currentForce > maxForce) {
      maxForce = currentForce;
    }

    // --- THE FRACTURE DETECTION ENGINE ---
    // Rule 1: We must be pulling with at least 0.5kg of force (Arming Threshold)
    // Rule 2: The current force must drop suddenly by 40% (Fracture Threshold)
    if (maxForce > armingThreshold && currentForce < (maxForce * fractureThreshold)) {
      isTesting = false; // This instantly cuts power to the motor logic
      Serial.println("--- FRACTURE DETECTED ---");
      Serial.print("MAX_FORCE:");
      Serial.println(maxForce);
    }
  }
}