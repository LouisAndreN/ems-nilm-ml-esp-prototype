#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <Preferences.h>


// ========= CALIBRATION SETTINGS ==========
float knownPower_P = 1250.0f;      // power reference (kettle)
float mainsVoltage_V = 100.0f;     // electrical network voltage
// ==============================================

Adafruit_ADS1115 ads;
Preferences prefs;

const float ADS_LSB = 4.096f / 32768.0f;  // GAIN_ONE
const int SAMPLES_CALIBRATION = 860;
const int SAMPLES_MONITORING = 200;

float calibrationFactor = 5.0f;
float offsetCurrent = 0.0f;

const char* PREF_KEY = "sct_factor";
const char* PREF_NAMESPACE = "sct_conf";
const float IGNORE_THRESHOLD_A = 0.001f;

float measure_vrms(int samples);
float measure_irms();
void autoCalibration();

unsigned long t0_start;  // temps de début du programme


void setup() {
  Serial.begin(115200);
  delay(500);
  
  Serial.println("\n");
  Serial.println("╔════════════════════════════════════════╗");
  Serial.println("║   ESP32 NILM Energy Monitor v1.0      ║");
  Serial.println("╚════════════════════════════════════════╝");
  Serial.println();
  
  // Init hardware
  if (!ads.begin()) {
    Serial.println("❌ ERROR: ADS1115 not found!");
    while (1) delay(1000);
  }
  
  ads.setGain(GAIN_ONE);
  ads.setDataRate(RATE_ADS1115_860SPS);
  Serial.println("✓ ADS1115 initialized");
  
  prefs.begin("nilm_config", false);
  Serial.println("✓ Preferences loaded");
  
  // Vérifier si calibration existe
  if (prefs.isKey("offset") && prefs.isKey("factor")) {
    offsetCurrent = prefs.getFloat("offset", 0);
    calibrationFactor = prefs.getFloat("factor", 0);
    
    Serial.println("\n--- Stored Calibration ---");
    Serial.print("Offset: ");
    Serial.print(offsetCurrent * 1000, 1);
    Serial.println(" mA");
    Serial.print("Factor: ");
    Serial.println(calibrationFactor, 2);
    
    Serial.println("\nRecalibrate? (y/n)");
    
    // Attendre 5 secondes pour réponse
    unsigned long t0 = millis();
    bool recalibrate = false;
    
    while (millis() - t0 < 5000) {
      if (Serial.available()) {
        char c = Serial.read();
        if (c == 'y' || c == 'Y') {
          recalibrate = true;
          break;
        } else if (c == 'n' || c == 'N') {
          break;
        }
      }
      delay(100);
    }
    
    if (recalibrate) {
      fullCalibration();
    } else {
      Serial.println("→ Using stored calibration\n");
    }
    
  } else {
    // Pas de calibration, forcer
    Serial.println("\n⚠️  No calibration found!");
    Serial.println("→ Starting calibration procedure...\n");
    fullCalibration();
  }
  
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║         SYSTEM READY                   ║");
  Serial.println("║   Monitoring every 1 second...         ║");
  Serial.println("╚════════════════════════════════════════╝\n");
  
  delay(2000);
}

void fullCalibration() {
  Serial.println("\n");
  Serial.println("╔════════════════════════════════════════╗");
  Serial.println("║      FULL CALIBRATION PROCEDURE        ║");
  Serial.println("╚════════════════════════════════════════╝");
  
  // Étape 1 : Offset
  calibrateOffset();
  
  delay(2000);
  
  // Étape 2 : Factor
  calibrateFactor();
  
  Serial.println("\n✓ Full calibration complete!\n");
}

float measure_irms_corrected() {
  // 1. Mesure rapide VRMS (200 samples ≈ 0.23 sec)
  float vrms = measure_vrms(SAMPLES_MONITORING);
  
  // 2. Conversion VRMS → courant brut
  float I_raw = vrms * calibrationFactor;
  
  // 3. Correction offset (soustraire bruit)
  float I_corrected = I_raw - offsetCurrent;
  
  // 4. Threshold (ignorer bruit résiduel <50 mA)
  if (I_corrected < NOISE_THRESHOLD_A) {
    I_corrected = 0;
  }
  
  return I_corrected;
}


void calibrateOffset() {
  Serial.println("\n========== OFFSET CALIBRATION ==========");
  Serial.println("Step 1: Disconnect ALL loads from circuit");
  Serial.println("        OR turn OFF circuit breaker");
  Serial.println("        (Keep SCT clipped on wire)");
  Serial.println("\nPress ENTER when ready...");
  
  // Attendre confirmation utilisateur
  while (!Serial.available()) delay(100);
  while (Serial.available()) Serial.read();  // Clear buffer
  
  Serial.println("\nMeasuring offset (10 samples)...");
  
  float sum = 0;
  float values[10];
  
  for (int i = 0; i < 10; i++) {
    float I = measure_irms_fast();  // Mesure rapide
    values[i] = I;
    sum += I;
    
    Serial.print("  #");
    Serial.print(i + 1);
    Serial.print(": ");
    Serial.print(I * 1000, 1);
    Serial.println(" mA");
    
    delay(1000);
  }
  
  // Moyenne
  offsetCurrent = sum / 10.0;
  
  // Écart-type (optionnel, pour info)
  float variance = 0;
  for (int i = 0; i < 10; i++) {
    float diff = values[i] - offsetCurrent;
    variance += diff * diff;
  }
  float stddev = sqrt(variance / 10.0);
  
  Serial.print("\n>>> Offset: ");
  Serial.print(offsetCurrent * 1000, 1);
  Serial.print(" mA ± ");
  Serial.print(stddev * 1000, 1);
  Serial.println(" mA");
  
  // Sauvegarder en mémoire persistante
  prefs.putFloat("offset", offsetCurrent);
  
  Serial.println(">>> Offset saved to memory");
  Serial.println("========================================\n");
}


void calibrateFactor() {
  Serial.println("\n========== FACTOR CALIBRATION ==========");
  Serial.print("Step 2: Connect load of ");
  Serial.print(knownPower_P, 0);
  Serial.print("W (");
  Serial.print(knownPower_P / mainsVoltage_V, 1);
  Serial.println("A)");
  Serial.println("        Turn ON the load");
  Serial.println("\nPress ENTER when load is running...");
  
  // Attendre confirmation
  while (!Serial.available()) delay(100);
  while (Serial.available()) Serial.read();
  
  Serial.println("\nWait 5 seconds for stabilization...");
  delay(5000);
  
  Serial.println("Measuring factor (5 samples, 1 sec each)...");
  
  float I_known = knownPower_P / mainsVoltage_V;
  const int NUM_SAMPLES = 5;
  float factors[NUM_SAMPLES];
  float sum = 0;
  
  for (int i = 0; i < NUM_SAMPLES; i++) {
    // Mesure VRMS précise (860 samples = 1 sec)
    float vrms = measure_vrms(SAMPLES_CALIBRATION);
    
    // Calcul facteur
    float factor = I_known / vrms;
    factors[i] = factor;
    sum += factor;
    
    Serial.print("  #");
    Serial.print(i + 1);
    Serial.print(": Vrms = ");
    Serial.print(vrms * 1000, 2);
    Serial.print(" mV → factor = ");
    Serial.println(factor, 2);
    
    delay(2000);  // Pause entre mesures
  }
  
  // Moyenne
  calibrationFactor = sum / NUM_SAMPLES;
  
  // Écart-type
  float variance = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    float diff = factors[i] - calibrationFactor;
    variance += diff * diff;
  }
  float stddev = sqrt(variance / NUM_SAMPLES);
  
  // Coefficient variation (%)
  float cv_percent = (stddev / calibrationFactor) * 100;
  
  Serial.print("\n>>> Calibration Factor: ");
  Serial.print(calibrationFactor, 2);
  Serial.print(" ± ");
  Serial.print(stddev, 2);
  Serial.print(" (±");
  Serial.print(cv_percent, 1);
  Serial.println("%)");
  
  // Validation qualité
  if (cv_percent > 5.0) {
    Serial.println("⚠️  WARNING: High variation (>5%)");
    Serial.println("    Check load stability, try again");
  } else {
    Serial.println("✓  Good calibration (variation <5%)");
  }
  
  // Sauvegarder
  prefs.putFloat("factor", calibrationFactor);
  
  Serial.println(">>> Factor saved to memory");
  Serial.println("========================================\n");
}

float measure_irms_with_time(unsigned long &time_ms) {
    unsigned long t_start = millis();   // début mesure
    float irms = measure_irms();
    unsigned long t_end = millis();     // fin mesure
    time_ms = ((t_start + t_end) / 2) - t0_start;  // temps relatif depuis début
    return irms;
}

void loop() {
  static unsigned long lastMillis = 0;
  unsigned long currentMillis = millis();
  
  // Update toutes les 1 seconde
  if (currentMillis - lastMillis >= 1000) {
    
    // Mesure corrigée (offset + threshold)
    float I = measure_irms_corrected();
    float P = I * mainsVoltage_V;
    
    // Affichage
    if (I > 0) {
      Serial.print("I = ");
      Serial.print(I, 2);
      Serial.print(" A | P = ");
      Serial.print(P, 0);
      Serial.println(" W");
    } else {
      Serial.println("OFF (no load detected)");
    }
    
    lastMillis = currentMillis;
  }
}


// ---- RMS CALC ----
float measure_vrms(int samples) {
  double sumsq = 0;
  for (int i = 0; i < samples; i++) {
    int16_t raw = ads.readADC_Differential_0_1();
    float v = raw * ADS_LSB;
    sumsq += v * v;
    delayMicroseconds(SAMPLE_DELAY_US);
  }
  return sqrt(sumsq / samples);
}


// ---- IRMS ----
float measure_irms() {
  float vrms = measure_vrms(SAMPLES);
  return vrms * calibrationFactor;
}


// ---- AUTO CALIBRATION ----
void autoCalibration() {
  Serial.println("=== Auto Calibration Started ===");

  Serial.print("Line voltage = ");
  Serial.print(mainsVoltage_V);
  Serial.println(" V");

  Serial.print("Expected I_known = ");
  Serial.print(knownCurrent_I, 6);
  Serial.println(" A");

  Serial.println("Measuring Vrms...");
  float vrms = measure_vrms(SAMPLES);

  Serial.print("Measured Vrms = ");
  Serial.print(vrms * 1000.0f, 4);
  Serial.println(" mV");

  calibrationFactor = knownCurrent_I / vrms;

  // Sauvegarde quand même dans NVS si tu veux réutiliser
  prefs.putFloat(PREF_KEY, calibrationFactor);

  Serial.print("New calibration factor = ");
  Serial.println(calibrationFactor, 6);
  Serial.println("=== Auto Calibration Done ===");
}
