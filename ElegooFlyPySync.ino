/*
  ElegooFlyPySync.ino — v3.0 (Analog Tripwire + Always-ON Laser + FlyAPI Tokens)
  ---------------------------------------------------------------------------
  Goal: Make FlyAPI reliably trigger when the beam is broken by a large object
        (e.g., pencil), even if the analog signal is noisy.

  This sketch:
    - Reads the LASER RECEIVER analog output on A5 (PIN_RX).
    - Optional receiver digital output on D2 (PIN_DO) (disabled by default).
    - Controls LASER EMITTER intensity with a POT on A0 (PIN_POT) driving PWM on D5 (PIN_LASER_PWM).
      * Laser is ALWAYS ON in firmware by enforcing PWM >= LASER_PWM_MIN.
      * For true fail-safe always-on during Arduino reset, add a Gate->+5V pull-up on the MOSFET.
    - Controls LIGHT LANE MOSFET gate on D9 (PIN_LANE_PWM) (optional).
    - Emits a SERIAL TRIGGER TOKEN to FlyAPI when a beam-break event is detected:
        default token is "0" (one line).
        legacy token can be enabled with OUTPUT LEGACY (sends "T").

  Trigger logic (covers your Examples A/B/C):
    A) Sudden drop: if lastRX - rx >= DROP_DELTA_COUNTS  (and lastRX >= PREMIN_DROP)
    B) Drop-to-zero: if rx <= ZERO_ABS_THRESHOLD        (and lastRX >= PREMIN_ZERO)
    C) Rising cross: if lastRX < RISE_ABS_THRESHOLD && rx >= RISE_ABS_THRESHOLD (optional)

  IMPORTANT WIRING NOTE (the #1 cause of "doesn't trigger"):
    - RX (receiver AO) MUST be on A5.
    - POT wiper MUST be on A0.
    If A5 is connected to the POT wiper, you'll only read the pot and blocking the beam won't change RX.

  Serial output:
    - Trigger token lines: "0" or "T"
    - Debug stream (ignored by updated FlyAPI): "A:<rx>"
    - Comments/acks (ignored by updated FlyAPI): start with "C:" or "ACK"/"ERR"

  Baud: 115200
*/

#include <Arduino.h>

// ---------------- Pins ----------------
const uint8_t PIN_RX        = A5;  // receiver AO (analog)
const uint8_t PIN_POT       = A0;  // potentiometer wiper (analog)
const uint8_t PIN_DO        = 2;   // optional receiver DO (digital), INPUT_PULLUP

const uint8_t PIN_LASER_PWM = 5;   // MOSFET gate for LASER emitter (PWM)
const uint8_t PIN_LANE_PWM  = 9;   // MOSFET gate for LIGHT lane (PWM)
const uint8_t PIN_TTL_OUT   = 8;   // TTL sync pulse output (optional external)
const uint8_t PIN_ONBOARD   = 13;  // onboard LED (visual indicator)

// ---------------- Output mode ----------------
enum OutputMode_t { OUTPUT_ZERO = 0, OUTPUT_LEGACY_T = 1 };
OutputMode_t gOutputMode = OUTPUT_ZERO;

// ---------------- Sampling ----------------
int gSampleHz = 500; // 200–1000 is plenty; higher = more serial spam if STREAM ON
unsigned long gSampleUs = 1000000UL / 500;
unsigned long tLastSampleUs = 0;

// ---------------- Trigger gates ----------------
int  gRefractoryMs = 1500; // shorter so you can test quickly
int  gIdleArmMs    = 300;  // lower so it re-arms quickly for bench testing
unsigned long tLastTriggerMs = 0;

// "Stable" definition for arming (|dRX| <= stableDelta counts)
int gStableDelta = 6;
unsigned long tStableStartMs = 0;

// ---------------- Example A/B/C parameters ----------------
// A) sudden edge (drop or rise). Default enables BOTH directions so it works with either sensor polarity.
bool gTrigDropEn = true;      // drop edge
int  gDropDeltaCounts = 40;    // counts
int  gPreMinDrop      = 150;   // require lastRX >= this for drop edge

bool gTrigRiseDeltaEn = true;  // rise edge
int  gRiseDeltaCounts = 40;    // counts
int  gPreMinRise      = 150;   // require lastRX >= this for rise edge

bool gTrigZeroEn = true;
int  gZeroAbsThreshold = 25; // B) drop-to-zero (absolute)
int  gPreMinZero       = 80; // require lastRX >= this to count for B

bool gTrigRiseEn = false;    // C) rising cross (off by default)
int  gRiseAbsThreshold = 900;

// Optional DO trigger (default OFF)
bool gUseDOTrigger = false;
const int DO_DEBOUNCE_MS = 3;
int lastDO = HIGH;
unsigned long tLastDOChangeMs = 0;

// ---------------- Laser control (Always ON) ----------------
bool gLaserEnable  = true;
int  gLaserPwmMin  = 40;   // enforce always-on in firmware (try 20–80)
int  gLaserPwmMax  = 255;

// ---------------- Lane control ----------------
bool gLaneEnable = false;
int  gLanePwm    = 255;

// ---------------- Streaming ----------------
bool gStreamAnalog = true; // prints A:<rx> (FlyAPI ignores if updated)
unsigned long gBootMuteUntilMs = 0;

// ---------------- Baseline + absolute threshold (robust for slow occlusions) ----------------
float baseline = 0.0f;
bool  baselineReady = false;
float gEmaAlpha = 0.995f;     // baseline drift tracking when intact (0.98..0.999)
int   gAbsDiffThreshold = 80; // counts away from baseline to count as "blocked" (tune 30..200)
int   gAbsDebounceMs = 10;    // must exceed threshold for this long
unsigned long tAbsStartMs = 0;

// ---------------- State ----------------
int lastRX = -1;

// ---------------- Helpers ----------------
static inline long clampl(long v, long a, long b){ return v < a ? a : (v > b ? b : v); }

static inline void recalcSampleInterval() {
  gSampleHz = (int)clampl(gSampleHz, 100, 2000);
  gSampleUs = (unsigned long)(1000000UL / (unsigned long)gSampleHz);
  tLastSampleUs = micros();
}

static inline int potToPwmAlwaysOn(int potRaw) {
  int minv = (int)clampl(gLaserPwmMin, 0, 255);
  int maxv = (int)clampl(gLaserPwmMax, 0, 255);
  if (maxv < minv) { int t = maxv; maxv = minv; minv = t; }

  long span = (long)(maxv - minv);
  long pwm  = (long)potRaw * span / 1023L + (long)minv;

  if (pwm < minv) pwm = minv;
  if (pwm > maxv) pwm = maxv;
  return (int)pwm;
}

static inline void applyLaserFromPot() {
  if (!gLaserEnable) {
    analogWrite(PIN_LASER_PWM, 0);
    return;
  }
  int potRaw = analogRead(PIN_POT);
  int pwm = potToPwmAlwaysOn(potRaw);
  analogWrite(PIN_LASER_PWM, pwm);
}

static inline void applyLane() {
  if (!gLaneEnable) { analogWrite(PIN_LANE_PWM, 0); return; }
  analogWrite(PIN_LANE_PWM, (int)clampl(gLanePwm, 0, 255));
}

static inline bool inRefractory(unsigned long nowMs) {
  return (nowMs - tLastTriggerMs) < (unsigned long)gRefractoryMs;
}

static inline bool isArmed(unsigned long nowMs) {
  if (gIdleArmMs <= 0) return true;
  if (tStableStartMs == 0) return false;
  return (nowMs - tStableStartMs) >= (unsigned long)gIdleArmMs;
}

static inline void pulseTTL(int ms=10) {
  digitalWrite(PIN_TTL_OUT, HIGH);
  digitalWrite(PIN_ONBOARD, HIGH);
  delay(ms);
  digitalWrite(PIN_TTL_OUT, LOW);
  digitalWrite(PIN_ONBOARD, LOW);
}

static inline void emitTriggerToken() {
  if (gOutputMode == OUTPUT_ZERO) Serial.println(F("0"));
  else Serial.println(F("T"));
  tLastTriggerMs = millis();
  pulseTTL(10);
}

static inline void streamRX(int rx) {
  if (!gStreamAnalog) return;
  if (gBootMuteUntilMs && millis() < gBootMuteUntilMs) return;
  Serial.print(F("A:"));
  Serial.println(rx);
}

#define CPRINT() Serial.print(F("C: "))

static inline void showStatus() {
  CPRINT(); Serial.print(F("HZ=")); Serial.print(gSampleHz);
  Serial.print(F(" REFRACT=")); Serial.print(gRefractoryMs);
  Serial.print(F(" ARM=")); Serial.print(gIdleArmMs);
  Serial.print(F(" STABLEDELTA=")); Serial.print(gStableDelta);
  Serial.print(F(" EMA=")); Serial.print(gEmaAlpha, 3);
  Serial.print(F(" ABSTHR=")); Serial.print(gAbsDiffThreshold);
  Serial.print(F(" ABSDEB=")); Serial.print(gAbsDebounceMs);

  Serial.print(F(" DROP_EN=")); Serial.print(gTrigDropEn ? F("ON") : F("OFF"));
  Serial.print(F(" DROPDELTA=")); Serial.print(gDropDeltaCounts);
  Serial.print(F(" PRE_DROP=")); Serial.print(gPreMinDrop);

  Serial.print(F(" RISEDELTA_EN=")); Serial.print(gTrigRiseDeltaEn ? F("ON") : F("OFF"));
  Serial.print(F(" RISEDELTA=")); Serial.print(gRiseDeltaCounts);
  Serial.print(F(" PRE_RISE=")); Serial.print(gPreMinRise);

  Serial.print(F(" ZERO_EN=")); Serial.print(gTrigZeroEn ? F("ON") : F("OFF"));
  Serial.print(F(" ZEROABS=")); Serial.print(gZeroAbsThreshold);
  Serial.print(F(" PRE_ZERO=")); Serial.print(gPreMinZero);

  Serial.print(F(" RISE_EN=")); Serial.print(gTrigRiseEn ? F("ON") : F("OFF"));
  Serial.print(F(" RISEABS=")); Serial.print(gRiseAbsThreshold);

  Serial.print(F(" DO=")); Serial.print(gUseDOTrigger ? F("ON") : F("OFF"));

  Serial.print(F(" LASER_MIN=")); Serial.print(gLaserPwmMin);
  Serial.print(F(" LASER_MAX=")); Serial.print(gLaserPwmMax);

  Serial.print(F(" LANE=")); Serial.print(gLaneEnable ? F("ON") : F("OFF"));
  Serial.print(F(" LANE_PWM=")); Serial.print(gLanePwm);

  Serial.print(F(" OUTPUT=")); Serial.print(gOutputMode == OUTPUT_ZERO ? F("0") : F("T"));
  Serial.print(F(" STREAM=")); Serial.println(gStreamAnalog ? F("ON") : F("OFF"));
}

static inline String trimBoth(String s){ s.trim(); return s; }
static inline String upper(String s){ s.toUpperCase(); return s; }

static inline bool parseOnOff(const String& s, bool &out) {
  String u = s; u.trim(); u.toUpperCase();
  if (u == "ON")  { out = true;  return true; }
  if (u == "OFF") { out = false; return true; }
  return false;
}

static inline void handleCommand(String line) {
  line = trimBoth(line);
  if (!line.length()) return;

  String u = upper(line);

  if (u == "HELP") {
    CPRINT(); Serial.println(F("Commands:"));
    CPRINT(); Serial.println(F("  STATUS | HELP"));
    CPRINT(); Serial.println(F("  STREAM ON|OFF"));
    CPRINT(); Serial.println(F("  OUTPUT TRIGGER|LEGACY"));
    CPRINT(); Serial.println(F("  SET HZ <100..2000>"));
    CPRINT(); Serial.println(F("  SET REFRACT <ms>"));
    CPRINT(); Serial.println(F("  SET ARM <ms>"));
    CPRINT(); Serial.println(F("  SET STABLEDELTA <cnt>"));
    CPRINT(); Serial.println(F("  SET EMA <0.90..0.999>"));
    CPRINT(); Serial.println(F("  SET ABSTHR <cnt> | SET ABSDEB <ms>"));
    CPRINT(); Serial.println(F("  SET DROPDELTA <cnt> | SET PRE_DROP <cnt> | TRIG DROP ON|OFF"));
    CPRINT(); Serial.println(F("  SET RISEDELTA <cnt> | SET PRE_RISE <cnt> | TRIG RISEDELTA ON|OFF"));
    CPRINT(); Serial.println(F("  SET ZEROABS <cnt>  | SET PRE_ZERO <cnt> | TRIG ZERO ON|OFF"));
    CPRINT(); Serial.println(F("  SET RISEABS <cnt>  | TRIG RISE ON|OFF"));
    CPRINT(); Serial.println(F("  DO ON|OFF"));
    CPRINT(); Serial.println(F("  SET PWM_MIN <0..255> | SET PWM_MAX <0..255>"));
    CPRINT(); Serial.println(F("  LANE ON|OFF | SET LANE <0..255>"));
    CPRINT(); Serial.println(F("  MARK START | MARK END | LIGHT ON | LIGHT OFF"));
    return;
  }
  if (u == "STATUS") { showStatus(); return; }

  if (u.startsWith("STREAM ")) {
    bool v; if (parseOnOff(u.substring(7), v)) { gStreamAnalog = v; CPRINT(); Serial.println(v ? F("STREAM=ON") : F("STREAM=OFF")); }
    else { CPRINT(); Serial.println(F("ERR STREAM ON|OFF")); }
    return;
  }

  if (u.startsWith("OUTPUT ")) {
    String arg = trimBoth(u.substring(7));
    if (arg == "TRIGGER") { gOutputMode = OUTPUT_ZERO; CPRINT(); Serial.println(F("OUTPUT=TRIGGER (0)")); }
    else if (arg == "LEGACY") { gOutputMode = OUTPUT_LEGACY_T; CPRINT(); Serial.println(F("OUTPUT=LEGACY (T)")); }
    else { CPRINT(); Serial.println(F("ERR OUTPUT TRIGGER|LEGACY")); }
    return;
  }

  if (u == "LIGHT ON")  { gLaneEnable = true;  applyLane(); Serial.println(F("ACK LIGHT ON")); return; }
  if (u == "LIGHT OFF") { gLaneEnable = false; applyLane(); Serial.println(F("ACK LIGHT OFF")); return; }

  if (u.startsWith("MARK")) { pulseTTL(5); Serial.println(String("ACK ") + line); return; }

  if (u.startsWith("DO ")) {
    bool v; if (parseOnOff(u.substring(3), v)) { gUseDOTrigger = v; CPRINT(); Serial.println(v ? F("DO=ON") : F("DO=OFF")); }
    else { CPRINT(); Serial.println(F("ERR DO ON|OFF")); }
    return;
  }

  if (u.startsWith("TRIG ")) {
    // TRIG DROP ON|OFF etc.
    String rest = trimBoth(u.substring(5));
    int sp = rest.indexOf(' ');
    if (sp < 0) { CPRINT(); Serial.println(F("ERR TRIG <DROP|ZERO|RISE|RISEDELTA> ON|OFF")); return; }
    String which = rest.substring(0, sp);
    String val = trimBoth(rest.substring(sp + 1));
    bool on;
    if (!parseOnOff(val, on)) { CPRINT(); Serial.println(F("ERR TRIG ... ON|OFF")); return; }
    if (which == "DROP") gTrigDropEn = on;
    else if (which == "ZERO") gTrigZeroEn = on;
    else if (which == "RISE") gTrigRiseEn = on;
    else if (which == "RISEDELTA") gTrigRiseDeltaEn = on;
    else { CPRINT(); Serial.println(F("ERR TRIG which (DROP|ZERO|RISE|RISEDELTA)")); return; }
    CPRINT(); Serial.print(F("TRIG ")); Serial.print(which); Serial.print(F("=")); Serial.println(on ? F("ON") : F("OFF"));
    return;
  }

  if (u.startsWith("SET ")) {
    String rest = trimBoth(u.substring(4));
    int sp = rest.indexOf(' ');
    if (sp < 0) { CPRINT(); Serial.println(F("ERR SET <KEY> <VALUE>")); return; }
    String key = rest.substring(0, sp);
    long n = trimBoth(rest.substring(sp + 1)).toInt();

    if (key == "HZ") { gSampleHz = (int)clampl(n, 100, 2000); recalcSampleInterval(); }
    else if (key == "REFRACT") { gRefractoryMs = (int)clampl(n, 0, 5000); }
    else if (key == "ARM") { gIdleArmMs = (int)clampl(n, 0, 5000); }
    else if (key == "STABLEDELTA") { gStableDelta = (int)clampl(n, 0, 200); }
    else if (key == "EMA") {
      // crude parse: accept integer 900..999 as 0.900..0.999
      long vv = clampl(n, 900, 999);
      gEmaAlpha = (float)vv / 1000.0f;
    }
    else if (key == "ABSTHR") { gAbsDiffThreshold = (int)clampl(n, 1, 1023); }
    else if (key == "ABSDEB") { gAbsDebounceMs = (int)clampl(n, 0, 500); }

    else if (key == "DROPDELTA") { gDropDeltaCounts = (int)clampl(n, 1, 1023); }
    else if (key == "PRE_DROP") { gPreMinDrop = (int)clampl(n, 0, 1023); }
    else if (key == "RISEDELTA") { gRiseDeltaCounts = (int)clampl(n, 1, 1023); }
    else if (key == "PRE_RISE") { gPreMinRise = (int)clampl(n, 0, 1023); }

    else if (key == "ZEROABS") { gZeroAbsThreshold = (int)clampl(n, 0, 1023); }
    else if (key == "PRE_ZERO") { gPreMinZero = (int)clampl(n, 0, 1023); }

    else if (key == "RISEABS") { gRiseAbsThreshold = (int)clampl(n, 0, 1023); }

    else if (key == "PWM_MIN") { gLaserPwmMin = (int)clampl(n, 0, 255); }
    else if (key == "PWM_MAX") { gLaserPwmMax = (int)clampl(n, 0, 255); }

    else if (key == "LANE") { gLanePwm = (int)clampl(n, 0, 255); gLaneEnable = true; applyLane(); }

    else { CPRINT(); Serial.println(F("ERR SET unknown key")); return; }

    CPRINT(); Serial.print(F("SET ")); Serial.print(key); Serial.print(F("=")); Serial.println(n);
    return;
  }

  Serial.println(String("ERR UNKNOWN ") + line);
}

void setup() {
  pinMode(PIN_TTL_OUT, OUTPUT);
  pinMode(PIN_ONBOARD, OUTPUT);
  digitalWrite(PIN_TTL_OUT, LOW);
  digitalWrite(PIN_ONBOARD, LOW);

  pinMode(PIN_LASER_PWM, OUTPUT);
  pinMode(PIN_LANE_PWM, OUTPUT);

  pinMode(PIN_RX, INPUT);
  pinMode(PIN_POT, INPUT);
  pinMode(PIN_DO, INPUT_PULLUP);

  Serial.begin(115200);
  delay(200);
  while (Serial.available()) Serial.read();
  gBootMuteUntilMs = millis() + 250;

  recalcSampleInterval();
  applyLaserFromPot();
  applyLane();

  CPRINT(); Serial.println(F("ElegooFlyPySync v3.0 ready @115200"));
  showStatus();
}

String lineBuf;

void loop() {
  // ---- parse commands from FlyAPI ----
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c != '\n') { lineBuf += c; continue; }
    String ln = lineBuf;
    lineBuf = "";
    ln.trim();
    if (ln.length()) handleCommand(ln);
  }

  // ---- keep outputs responsive ----
  applyLaserFromPot();
  applyLane();

  // ---- sample receiver ----
  unsigned long nowUs = micros();
  if ((long)(nowUs - tLastSampleUs) < (long)gSampleUs) return;
  tLastSampleUs += gSampleUs;

  unsigned long nowMs = millis();
  int rx = analogRead(PIN_RX);

  // stream receiver value for debugging / calibration
  streamRX(rx);

  // initialize lastRX + baseline
  if (lastRX < 0) {
    lastRX = rx;
    baseline = (float)rx;
    baselineReady = true;
    tStableStartMs = nowMs;
    return;
  }

  // stability tracker for arming
  int dv = rx - lastRX;
  if (abs(dv) <= gStableDelta) {
    if (tStableStartMs == 0) tStableStartMs = nowMs;
  } else {
    tStableStartMs = 0;
  }

  // ---- Baseline drift tracking (only when "intact") ----
  if (!baselineReady) { baseline = (float)rx; baselineReady = true; }
  float diffAbs = fabsf((float)rx - baseline);
  bool nearBaseline = diffAbs < (float)max(5, gAbsDiffThreshold / 4);
  bool stableNow = (abs(dv) <= gStableDelta);
  // Update baseline only when stable and near baseline (avoids learning occlusions)
  if (stableNow && nearBaseline) {
    baseline = gEmaAlpha * baseline + (1.0f - gEmaAlpha) * (float)rx;
  }

  // Absolute threshold gate (robust for slow occlusions)
  bool trigABS = false;
  if (diffAbs >= (float)gAbsDiffThreshold) {
    if (tAbsStartMs == 0) tAbsStartMs = nowMs;
    if ((nowMs - tAbsStartMs) >= (unsigned long)gAbsDebounceMs) trigABS = true;
  } else {
    tAbsStartMs = 0;
  }

  bool ok = !inRefractory(nowMs) && isArmed(nowMs);
  if (!ok) trigABS = false;

  // Example A: sudden DROP edge
  bool trigA = false;
  if (ok && gTrigDropEn) {
    int drop = lastRX - rx;
    if (lastRX >= gPreMinDrop && drop >= gDropDeltaCounts) trigA = true;
  }

  // Extra robustness: sudden RISE edge (some receivers increase when beam is blocked)
  bool trigA2 = false;
  if (ok && gTrigRiseDeltaEn) {
    int rise = rx - lastRX;
    if (lastRX >= gPreMinRise && rise >= gRiseDeltaCounts) trigA2 = true;
  }

  // Example B: drop-to-zero
  bool trigB = false;
  if (ok && gTrigZeroEn) {
    if (rx <= gZeroAbsThreshold && lastRX >= gPreMinZero) trigB = true;
  }

  // Example C: rising cross
  bool trigC = false;
  if (ok && gTrigRiseEn) {
    if (lastRX < gRiseAbsThreshold && rx >= gRiseAbsThreshold) trigC = true;
  }

  // Optional DO trigger
  bool trigDO = false;
  if (ok && gUseDOTrigger) {
    int doLevel = digitalRead(PIN_DO);
    if (doLevel != lastDO) { lastDO = doLevel; tLastDOChangeMs = nowMs; }
    if (lastDO == LOW && (nowMs - tLastDOChangeMs) >= (unsigned long)DO_DEBOUNCE_MS) trigDO = true;
  }

  if (trigABS || trigA || trigA2 || trigB || trigC || trigDO) {
    emitTriggerToken();
    // force re-arm (needs stability again)
    tStableStartMs = 0;
  }

  lastRX = rx;
}
