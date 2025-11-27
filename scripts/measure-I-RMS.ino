#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;

// Configuration
const int SAMPLES = 2000;           // sample number for RMS
const int SAMPLE_DELAY_US = 1200;   // sampling frequency (in ms)
const float ADS_LSB = 2.048f / 32768.0f; // 62.5 µV (GAIN_ONE)
const float CT_A_PER_V = 5.0f;     // calibration SCT-013-005 : 5A/1V
// const float IGNORE_THRESHOLD_A = 0.01f; // ignore < 10 mA

// Baseline measure (noise) at start
float baseline_vrms = 0.0f;

void setup() {
  Serial.begin(115200);
  delay(300);
  // Start RMS

  if (!ads.begin()) {
    Serial.println("ERROR: ADS1115 undetected !");
    while (1) delay(1000);
  }
  ads.setGain(GAIN_ONE); // ±2.048 V

  // Measure baseline (CT unpplugged)
  // Short measure -> (SAMPLES / 4)
  baseline_vrms = measure_vrms(SAMPLES / 4);
  Serial.print("Baseline Vrms (bruit) = ");
  Serial.print(baseline_vrms * 1000.0f, 3);
  Serial.println(" mV");
  Serial.println("----");
}

float measure_vrms(int samples) {
  // Differential between A0 and A1
  double sumsq = 0.0;
  for (int i = 0; i < samples; ++i) {
    int16_t raw = ads.readADC_Differential_0_1();
    float v = raw * ADS_LSB;
    sumsq += (double)v * (double)v;
    delayMicroseconds(SAMPLE_DELAY_US);
  }
  float meanSq = (float)(sumsq / (double)samples);
  return sqrt(meanSq); // VRMS
}

void loop() {
  // Long measure for high precision
  float measured_vrms = measure_vrms(SAMPLES);

  // Quadratic correction of noise : Vrms_corrected = sqrt(max(0, Vrms^2 - Vrms_baseline^2))
  float v2 = measured_vrms * measured_vrms;
  float b2 = baseline_vrms * baseline_vrms;
  float corrected_vrms = 0.0f;
  if (v2 > b2) corrected_vrms = sqrt(v2 - b2);

  // Conversion in RMS current
  float current_rms = corrected_vrms * CT_A_PER_V;

  // Plot results
  Serial.print("Measured Vrms: ");
  Serial.print(measured_vrms * 1000.0f, 4); Serial.print(" mV  |  ");
  Serial.print("Corrected Vrms: ");
  Serial.print(corrected_vrms * 1000.0f, 4); Serial.print(" mV  |  ");
  Serial.print("I_rms: ");
  Serial.print(current_rms, 6); Serial.print(" A");

  // Hide residual noise
  if (current_rms < IGNORE_THRESHOLD_A) {
    Serial.print("  (below threshold)");
  }
  Serial.println();

  delay(1000);
}
