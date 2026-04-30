#include <Wire.h>
//#include <Servo.h>
#include <Arduino.h>
#include "IMU_Fusion_SYC.h"
#include <ESP32Servo.h>
IMU imu(Wire);

#define ENCODER_PIN 27
#define MOTOR_PWM_PIN 17
#define DIR1_PIN 12
#define DIR2_PIN 14
#define STEERING_SERVO_PIN 18

#define PWM_FREQ 20000
#define PWM_RES 8
const int pwmChannel = 5;

volatile long encoder_ticks = 0;
const float TICKS_PER_METER = 6998;
float alpha = 0.15f; // filter strength for the encoder calculations
float yaw = 0.0;  // current yaw
float v = 0.0;    // speed

Servo steeringServo;

void setup() {
  Serial.begin(115200);
  pinMode(DIR1_PIN, OUTPUT);
  pinMode(DIR2_PIN, OUTPUT);
  setupMPU6050();

  pinMode(ENCODER_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN), countEncoder, RISING);

  ledcSetup(pwmChannel, PWM_FREQ, PWM_RES);
  ledcAttachPin(MOTOR_PWM_PIN, pwmChannel);
  steeringServo.attach(18);
}

void loop() {
  digitalWrite(DIR1_PIN, HIGH);
  digitalWrite(DIR2_PIN, LOW);
  readIMU();
  computeSpeed();

  // Send sensor data to Raspberry Pi
  Serial.print("DATA:");
  Serial.print(yaw); Serial.print(",");
  Serial.println(v);

  // Receive motor command [steering_angle_deg, pwm_motor]
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    int commaIndex = line.indexOf(',');
    if (commaIndex > 0) {
      float steering_angle_deg = line.substring(0, commaIndex).toFloat();
      int pwm_motor = line.substring(commaIndex + 1).toInt();

      steeringServo.write(constrain(steering_angle_deg, 40, 140));
      ledcWrite(pwmChannel, constrain(pwm_motor, 0, 255));
    }
  }
  delay(20);
}

void countEncoder() { encoder_ticks++; }

void setupMPU6050() {
  Wire.begin();                // Uses 21 = SDA, 22 = SCL on ESP32
  delay(1000);
  imu.begin(CHOOSE_MPU6050);   // Initialize MPU6050
  delay(1000);
  imu.MPU6050_CalcGyroOffsets();
}

void readIMU() {
  imu.Calculate();   // Update fused angles
  yaw=imu.getAngleZ();
}

void computeSpeed() {
  static long last_ticks = 0;
  static unsigned long last_time = 0;
  noInterrupts();
  long ticks = encoder_ticks;
  interrupts();
  unsigned long t = millis();
  long dticks = ticks - last_ticks;
  float distance = (float)dticks / TICKS_PER_METER;
  float dt = (t - last_time) / 1000.0;
  v = alpha * (distance / dt) + (1 - alpha) * v;   
  last_ticks = ticks;
  last_time = t;
}
