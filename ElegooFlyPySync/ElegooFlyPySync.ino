#include <Arduino.h>

const uint8_t PIN_BEAM      = A5;   // receiver WHITE (digital)
const uint8_t PIN_POT       = A0;   // pot wiper
const uint8_t PIN_LASER_PWM = 5;    // MOSFET gate PWM
const uint8_t PIN_LED       = 13;   // blip on trigger (UNO yellow "L" LED)

const char TRIG_TOKEN = 'T';        // <-- PATCH: FlyAPI currently expects "T"

const uint16_t SAMPLE_HZ       = 4000; // fast polling
const uint16_t DEBOUNCE_US     = 800;  // broken must persist
const uint16_t REARM_MS        = 10;   // intact must persist to re-arm
const uint16_t REFRACTORY_MS   = 150;  // minimum gap

// IMPORTANT: keep beam bright during debug; lower later if needed.
const uint8_t LASER_PWM_MIN = 180;  // raise/lower later
const uint8_t LASER_PWM_MAX = 255;

unsigned long tLastSampleUs = 0;
unsigned long brokenSinceUs = 0;
unsigned long intactSinceMs = 0;
unsigned long tLastTrigMs   = 0;

bool armed = true;

static inline void updateLaserPWM() {
  int pot = analogRead(PIN_POT); // 0..1023
  int pwm = map(pot, 0, 1023, LASER_PWM_MIN, LASER_PWM_MAX);
  pwm = constrain(pwm, LASER_PWM_MIN, LASER_PWM_MAX);
  analogWrite(PIN_LASER_PWM, (uint8_t)pwm);
}

static inline void emitTrigger() {
  Serial.println(TRIG_TOKEN);     // <-- emits "T" for FlyAPI
  tLastTrigMs = millis();

  digitalWrite(PIN_LED, HIGH);    // <-- PATCH: visible blink
  delay(100);
  digitalWrite(PIN_LED, LOW);
}

void setup() {
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  pinMode(PIN_BEAM, INPUT_PULLUP);  // open-collector → pull-up required
  pinMode(PIN_LASER_PWM, OUTPUT);

  Serial.begin(115200);
  delay(200);

  // Optional debug banners (safe if FlyAPI ignores lines starting with "C:")
  Serial.println(F("C: FlyPy breakbeam trigger ready"));
  Serial.println(F("C: Emits 'T' on beam break"));

  tLastSampleUs = micros();
  intactSinceMs = millis();
}

void loop() {
  unsigned long nowMs = millis();
  updateLaserPWM();

  // sampling cadence
  unsigned long nowUs = micros();
  const unsigned long dtUs = 1000000UL / SAMPLE_HZ;
  if ((long)(nowUs - tLastSampleUs) < (long)dtUs) return;
  tLastSampleUs += dtUs;

  bool broken = (digitalRead(PIN_BEAM) == LOW);  // LOW = beam broken
  bool inRefractory = (nowMs - tLastTrigMs) < REFRACTORY_MS;

  if (broken) {
    intactSinceMs = 0;
    if (brokenSinceUs == 0) brokenSinceUs = nowUs;

    if (armed && !inRefractory && (unsigned long)(nowUs - brokenSinceUs) >= DEBOUNCE_US) {
      emitTrigger();
      armed = false;          // lock out until beam is intact again
      brokenSinceUs = 0;
    }
  } else { // intact
    brokenSinceUs = 0;
    if (intactSinceMs == 0) intactSinceMs = nowMs;

    if (!armed && (nowMs - intactSinceMs) >= REARM_MS) {
      armed = true;           // re-arm after stable intact
    }
  }
}