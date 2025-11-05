/*
  ElegooFlyPySync — v2.6 (Zero-token Trigger Mode for FlyAPI v1.35.7)
  - DEFAULT: OUTPUT TRIGGER mode — emits '0' over the UNO's serial COM
    when the ANALOG beam dip meets THR + DEBOUNCE + AREA and ARM is satisfied.
    (Matches FlyAPI v1.35.7 default token filter.)
  - LEGACY mode available: OUTPUT LEGACY (emits 'T' instead of '0').
  - OUTPUT STATE mode: prints "state:ON"/"state:OFF" (FlyAPI ignores these).
  - Streams analog as "A:<value>" at SAMPLE_HZ (toggle with STREAM).
  - Robust detection: EMA baseline that freezes during dips, debounce, and area-gate (depth×time).
  - Live commands: SET THR/EMA/DEBOUNCE/REFRACT/ARM/HZ/AREA/LEDMS,
                   MODE, STREAM, DO, STATUS, HELP, BASELINE RESET,
                   AUTOTHR [ms], MARK START, MARK END, PING,
                   OUTPUT STATE|TRIGGER|LEGACY
  - Replies to commands start with "C:" (FlyAPI ignores them while listening for A:/token)

  Wiring:
    Receiver VCC -> 5V
    Receiver GND -> GND
    Receiver AO  -> A0  (ANALOG path used for trigger → serial '0' to FlyAPI via COM)
    (Optional) Receiver DO -> D2 (active LOW typical; default OFF in firmware)
    Sync LED     -> D13 (MARK START/END and optional blip on events)

  Baud: 115200
*/

#include <Arduino.h>
#include <math.h>

// ---------------- Runtime-tunable parameters (defaults) ----------------
int   gDropThreshold      = 40;     // counts below baseline to consider a dip (30–80 typical)
int   gDebounceMs         = 15;     // ms dip must persist
int   gRefractoryMs       = 3000;   // ms minimum between trigger events (default 3 s)
int   gIdleArmMs          = 3000;   // ms near-baseline before next trigger allowed (default 3 s)
float gEmaAlpha           = 0.95f;  // 0.90–0.9999 (lower = faster drift tracking)

int   gSampleHz           = 1000;   // 100–2000
unsigned long gSampleUs   = 1000000UL / 1000;

bool  gAnalogTriggerEn    = true;   // use ANALOG dip → trigger (default ON)
bool  gUseDOTrigger       = false;  // digital DO low → trigger (default OFF; analog path is cleaner)
bool  gStreamAnalog       = true;   // print "A:<value>" lines

int   gMinAreaCountMs     = 1200;   // dip "area" threshold in counts*ms (600–2000 good range)
int   gLedPulseMs         = 20;     // LED pulse width on events (ms)

// ---------------- Output mode ----------------
enum OutputMode_t { OUTPUT_ZERO = 0, OUTPUT_STATE = 1, OUTPUT_LEGACY_T = 2 };
OutputMode_t gOutputMode = OUTPUT_ZERO;  // << DEFAULT: emits '0' for FlyAPI v1.35.7

// ---------------- Pins ----------------
const int PIN_A    = A0;
const int PIN_DO   = 2;
const int PIN_SYNC = 13;

// ---------------- Digital DO debounce ----------------
const int DO_DEBOUNCE_MS  = 3;

// ---------------- Internal state ----------------
float baseline               = NAN;
unsigned long tLastSampleUs  = 0;

unsigned long tDipStartMs    = 0;
unsigned long tLastTriggerMs = 0;   // used by TRIGGER modes and LED timing
unsigned long tIdleStartMs   = 0;

int lastDO = HIGH;
unsigned long tLastDOChange  = 0;

long dropArea_counts_us      = 0;   // integrates drop * sample_us (counts*µs)
unsigned long tLedPulseStartMs = 0; // non-blocking LED auto-off

// STATE mode: -1 unknown, 0=OFF (broken), 1=ON (intact)
int gLastState = -1;

// Replies prefix (ignored by FlyAPI)
#define C_PREFIX "C: "

// ---------------- Helpers ----------------
static inline float clampf(float v, float a, float b){ return v < a ? a : (v > b ? b : v); }
static inline long  clampl(long v, long a, long b){ return v < a ? a : (v > b ? b : v); }

inline void printA(int v) {
  if (gStreamAnalog) { Serial.print("A:"); Serial.println(v); }
}

inline void blipLed() {
  if (gLedPulseMs <= 0) return;
  digitalWrite(PIN_SYNC, HIGH);
  tLedPulseStartMs = millis();
}

// Emit the serial token for a confirmed "beam broken" event.
//  - OUTPUT_ZERO    → prints "0"
//  - OUTPUT_LEGACY_T→ prints "T"
//  - OUTPUT_STATE   → (not used here; state mode prints "state:*")
inline void emitBrokenToken() {
  if (gOutputMode == OUTPUT_ZERO) {
    Serial.println("0");
    tLastTriggerMs = millis();
    blipLed();
  } else if (gOutputMode == OUTPUT_LEGACY_T) {
    Serial.println("T");
    tLastTriggerMs = millis();
    blipLed();
  }
}

inline void recalcSampleInterval() {
  gSampleHz = (int)clampl(gSampleHz, 100, 2000);
  gSampleUs = (unsigned long)(1000000UL / (unsigned long)gSampleHz);
  tLastSampleUs = micros(); // reschedule from now
}

void showStatus() {
  Serial.print(C_PREFIX "THR="); Serial.print(gDropThreshold);
  Serial.print(" EMA="); Serial.print(gEmaAlpha, 4);
  Serial.print(" DEBOUNCE="); Serial.print(gDebounceMs);
  Serial.print(" REFRACT="); Serial.print(gRefractoryMs);
  Serial.print(" ARM="); Serial.print(gIdleArmMs);
  Serial.print(" HZ="); Serial.print(gSampleHz);
  Serial.print(" AREA="); Serial.print(gMinAreaCountMs);
  Serial.print(" LEDMS="); Serial.print(gLedPulseMs);
  Serial.print(" MODE=");
  if (gAnalogTriggerEn && gUseDOTrigger) Serial.print("BOTH");
  else if (gAnalogTriggerEn)             Serial.print("ANALOG");
  else if (gUseDOTrigger)                Serial.print("DIGITAL");
  else                                   Serial.print("NONE");
  Serial.print(" OUTPUT=");
  if      (gOutputMode == OUTPUT_ZERO)     Serial.print("ZERO");
  else if (gOutputMode == OUTPUT_STATE)    Serial.print("STATE");
  else if (gOutputMode == OUTPUT_LEGACY_T) Serial.print("LEGACY");
  Serial.print(" STREAM="); Serial.println(gStreamAnalog ? "ON" : "OFF");
}

void showHelp() {
  Serial.println(C_PREFIX "Commands:");
  Serial.println(C_PREFIX "  SET THR <int>         (0..1023) analog drop threshold");
  Serial.println(C_PREFIX "  SET EMA <float>       (0.90..0.9999) baseline smoothing");
  Serial.println(C_PREFIX "  SET DEBOUNCE <ms>     (0..100)");
  Serial.println(C_PREFIX "  SET REFRACT <ms>      (0..5000) minimum gap between triggers");
  Serial.println(C_PREFIX "  SET ARM <ms>          (0..5000) near-baseline time before next trigger");
  Serial.println(C_PREFIX "  SET HZ <int>          (100..2000) sample/stream rate");
  Serial.println(C_PREFIX "  SET AREA <cnt*ms>     (0..50000) dip area gate (counts*ms)");
  Serial.println(C_PREFIX "  SET LEDMS <ms>        (0..1000) LED pulse on events");
  Serial.println(C_PREFIX "  MODE ANALOG|DIGITAL|BOTH|NONE");
  Serial.println(C_PREFIX "  STREAM ON|OFF");
  Serial.println(C_PREFIX "  DO ON|OFF");
  Serial.println(C_PREFIX "  OUTPUT STATE|TRIGGER|LEGACY");
  Serial.println(C_PREFIX "     TRIGGER → emits '0' (FlyAPI v1.35.7 default)");
  Serial.println(C_PREFIX "     LEGACY  → emits 'T' (back-compat)");
  Serial.println(C_PREFIX "  BASELINE RESET");
  Serial.println(C_PREFIX "  AUTOTHR [ms]          (500..10000) auto-set THR = max(20, 1.5×pp) @ idle");
  Serial.println(C_PREFIX "  STATUS | HELP | MARK START | MARK END | PING");
  Serial.println(C_PREFIX "Notes: Replies start with 'C:' so FlyAPI ignores them.");
}

String trimBoth(const String& s) { String t = s; t.trim(); return t; }
String upper(const String& s)    { String t = s; t.toUpperCase(); return t; }

void firstToken(const String& line, String& tok, String& rest) {
  int sp = line.indexOf(' ');
  if (sp < 0) { tok = line; rest = ""; }
  else { tok = line.substring(0, sp); rest = trimBoth(line.substring(sp + 1)); }
}

bool parseOnOff(const String& s, bool& out) {
  String u = upper(s);
  if (u == "ON")  { out = true;  return true; }
  if (u == "OFF") { out = false; return true; }
  return false;
}

// --- Auto-threshold calibration ---
void autoThreshold(unsigned long duration_ms) {
  duration_ms = (unsigned long)clampl((long)duration_ms, 500, 10000);

  // Save state and disable triggers during scan
  bool prevAnalog = gAnalogTriggerEn;
  bool prevDO     = gUseDOTrigger;
  gAnalogTriggerEn = false;
  gUseDOTrigger    = false;

  Serial.print(C_PREFIX "AUTOTHR start, ms="); Serial.println(duration_ms);
  Serial.println(C_PREFIX "Ensure beam is INTACT (no occlusions) during scan...");

  int minV = 1023, maxV = 0;
  unsigned long startMs = millis();

  unsigned long nextUs = micros();
  while ((millis() - startMs) < duration_ms) {
    unsigned long nowUs = micros();
    if ((long)(nowUs - nextUs) >= 0) {
      nextUs += gSampleUs;
      int v = analogRead(PIN_A);
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
      printA(v);
    }
    if (Serial.available()) {
      String line = Serial.readStringUntil('\n'); line.trim();
      if (upper(line) == "ABORT") {
        Serial.println(C_PREFIX "AUTOTHR aborted.");
        gAnalogTriggerEn = prevAnalog; gUseDOTrigger = prevDO;
        return;
      }
    }
  }

  int pp = maxV - minV;
  int thr = (int)ceil(1.5f * (float)pp);
  if (thr < 20) thr = 20;
  if (thr > 1023) thr = 1023;

  gDropThreshold = thr;

  Serial.print(C_PREFIX "Scan done. min="); Serial.print(minV);
  Serial.print(" max="); Serial.print(maxV);
  Serial.print(" pp="); Serial.print(pp);
  Serial.print(" -> THR="); Serial.println(gDropThreshold);

  gAnalogTriggerEn = prevAnalog;
  gUseDOTrigger    = prevDO;
}

void handleCommand(const String& lineRaw) {
  String line = trimBoth(lineRaw);
  if (!line.length()) return;

  String L = upper(line);
  if (L == "HELP")    { showHelp(); return; }
  if (L == "STATUS")  { showStatus(); return; }
  if (L == "BASELINE RESET") { baseline = NAN; Serial.println(C_PREFIX "Baseline reset"); return; }
  if (L == "PING")    { Serial.println("PONG"); return; }
  if (L == "MARK START") { digitalWrite(PIN_SYNC, HIGH);  tLedPulseStartMs = 0; return; }
  if (L == "MARK END")   { digitalWrite(PIN_SYNC, LOW);   tLedPulseStartMs = 0; return; }

  // AUTOTHR [ms]
  if (L.startsWith("AUTOTHR")) {
    unsigned long dur = 2000;
    if (L.length() > 7) {
      String arg = trimBoth(line.substring(7));
      if (arg.length()) {
        long v = arg.toInt();
        if (v > 0) dur = (unsigned long)v;
      }
    }
    autoThreshold(dur);
    return;
  }

  // MODE <...>
  if (L.startsWith("MODE ")) {
    String arg = upper(trimBoth(line.substring(5)));
    if      (arg == "ANALOG")  { gAnalogTriggerEn = true;  gUseDOTrigger = false; Serial.println(C_PREFIX "MODE=ANALOG"); }
    else if (arg == "DIGITAL") { gAnalogTriggerEn = false; gUseDOTrigger = true;  Serial.println(C_PREFIX "MODE=DIGITAL"); }
    else if (arg == "BOTH")    { gAnalogTriggerEn = true;  gUseDOTrigger = true;  Serial.println(C_PREFIX "MODE=BOTH"); }
    else if (arg == "NONE")    { gAnalogTriggerEn = false; gUseDOTrigger = false; Serial.println(C_PREFIX "MODE=NONE"); }
    else { Serial.println(C_PREFIX "ERR mode (ANALOG|DIGITAL|BOTH|NONE)"); }
    return;
  }

  // STREAM <ON|OFF>
  if (L.startsWith("STREAM ")) {
    String arg = trimBoth(line.substring(7)); bool v;
    if (parseOnOff(arg, v)) { gStreamAnalog = v; Serial.print(C_PREFIX "STREAM="); Serial.println(v ? "ON" : "OFF"); }
    else Serial.println(C_PREFIX "ERR STREAM ON|OFF");
    return;
  }

  // DO <ON|OFF>
  if (L.startsWith("DO ")) {
    String arg = trimBoth(line.substring(3)); bool v;
    if (parseOnOff(arg, v)) { gUseDOTrigger = v; Serial.print(C_PREFIX "DO="); Serial.println(v ? "ON" : "OFF"); }
    else Serial.println(C_PREFIX "ERR DO ON|OFF");
    return;
  }

  // OUTPUT STATE|TRIGGER|LEGACY
  if (L.startsWith("OUTPUT ")) {
    String arg = upper(trimBoth(line.substring(7)));
    if      (arg == "STATE")   { gOutputMode = OUTPUT_STATE;    Serial.println(C_PREFIX "OUTPUT=STATE"); }
    else if (arg == "TRIGGER") { gOutputMode = OUTPUT_ZERO;     Serial.println(C_PREFIX "OUTPUT=TRIGGER (ZERO)"); }
    else if (arg == "LEGACY")  { gOutputMode = OUTPUT_LEGACY_T; Serial.println(C_PREFIX "OUTPUT=LEGACY (T)"); }
    else { Serial.println(C_PREFIX "ERR OUTPUT STATE|TRIGGER|LEGACY"); }
    return;
  }

  // SET <KEY> <VAL>
  if (L.startsWith("SET ")) {
    String rest; String tok;
    firstToken(trimBoth(line.substring(4)), tok, rest);
    String key = upper(tok);
    if (!rest.length()) { Serial.println(C_PREFIX "ERR SET needs value"); return; }

    if (key == "THR" || key == "THRESH" || key == "THRESHOLD") {
      gDropThreshold = clampl(rest.toInt(), 0, 1023);
      Serial.print(C_PREFIX "THR="); Serial.println(gDropThreshold);
      return;
    }
    if (key == "EMA") {
      float v = rest.toFloat();
      gEmaAlpha = clampf(v, 0.90f, 0.9999f);
      Serial.print(C_PREFIX "EMA="); Serial.println(gEmaAlpha, 4);
      return;
    }
    if (key == "DEBOUNCE") {
      gDebounceMs = clampl(rest.toInt(), 0, 100);
      Serial.print(C_PREFIX "DEBOUNCE="); Serial.println(gDebounceMs);
      return;
    }
    if (key == "REFRACT" || key == "REFRACTORY") {
      gRefractoryMs = clampl(rest.toInt(), 0, 5000);
      Serial.print(C_PREFIX "REFRACT="); Serial.println(gRefractoryMs);
      return;
    }
    if (key == "ARM" || key == "IDLE" || key == "IDLEARM") {
      gIdleArmMs = clampl(rest.toInt(), 0, 5000);
      Serial.print(C_PREFIX "ARM="); Serial.println(gIdleArmMs);
      return;
    }
    if (key == "HZ" || key == "SAMPLEHZ") {
      gSampleHz = rest.toInt();
      recalcSampleInterval();
      Serial.print(C_PREFIX "HZ="); Serial.println(gSampleHz);
      return;
    }
    if (key == "AREA") {
      gMinAreaCountMs = clampl(rest.toInt(), 0, 50000);
      Serial.print(C_PREFIX "AREA="); Serial.println(gMinAreaCountMs);
      return;
    }
    if (key == "LEDMS") {
      gLedPulseMs = clampl(rest.toInt(), 0, 1000);
      Serial.print(C_PREFIX "LEDMS="); Serial.println(gLedPulseMs);
      return;
    }

    Serial.println(C_PREFIX "ERR unknown key (THR|EMA|DEBOUNCE|REFRACT|ARM|HZ|AREA|LEDMS)");
    return;
  }

  Serial.println(C_PREFIX "ERR unknown command (try HELP)");
}

void setup() {
  pinMode(PIN_SYNC, OUTPUT);
  digitalWrite(PIN_SYNC, LOW);

  pinMode(PIN_A, INPUT);
  pinMode(PIN_DO, INPUT_PULLUP); // safe if unconnected

  Serial.begin(115200);
  recalcSampleInterval();
}

void loop() {
  // ---- Serial command channel ----
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length()) handleCommand(line);
  }

  // ---- Sample timing ----
  unsigned long nowUs = micros();
  if ((long)(nowUs - tLastSampleUs) < (long)gSampleUs) return;
  tLastSampleUs += gSampleUs;

  // ---- Read sensors ----
  int v = analogRead(PIN_A);     // 0..1023
  int doLevel = digitalRead(PIN_DO);
  unsigned long nowMs = millis();

  // Stream analog
  printA(v);

  // Baseline handling (freeze during dips)
  if (isnan(baseline)) baseline = (float)v;

  // Compute drop using previous baseline
  int drop = (int)(baseline - (float)v); // positive = darker (fly blocking)
  bool nearBaseline = (drop < (gDropThreshold / 4));

  // Only let the baseline drift when we are near baseline (no occlusion)
  if (nearBaseline) {
    baseline = gEmaAlpha * baseline + (1.0f - gEmaAlpha) * (float)v;
  }

  // Track "near baseline" time (used to (re)arm and for STATE ON)
  if (nearBaseline) {
    if (tIdleStartMs == 0) tIdleStartMs = nowMs;
  } else {
    tIdleStartMs = 0;
  }

  // Area gate: integrate drop * sample_time (counts*µs)
  if (drop > 0) {
    long incr = (long)drop * (long)gSampleUs; // counts*µs
    long maxSafe = 2000000000L;
    if (dropArea_counts_us <= maxSafe - incr) dropArea_counts_us += incr;
    else dropArea_counts_us = maxSafe;
  }
  // Reset area when back near baseline
  if (nearBaseline) {
    dropArea_counts_us = 0;
  }

  // Debounce timer for dips
  if (drop >= gDropThreshold) {
    if (tDipStartMs == 0) tDipStartMs = nowMs;
  } else {
    tDipStartMs = 0;
  }

  // Optional DO path timing (still available; default OFF)
  if (gUseDOTrigger) {
    if (doLevel != lastDO) { lastDO = doLevel; tLastDOChange = nowMs; }
  }

  // ---------------- OUTPUT LOGIC ----------------
  if (gOutputMode == OUTPUT_STATE) {
    // Decide broken/intact with same robustness as trigger mode
    bool debounceOK = (tDipStartMs != 0) && ((nowMs - tDipStartMs) >= (unsigned long)gDebounceMs);
    long areaThresh_counts_us = (long)gMinAreaCountMs * 1000L;
    bool areaOK = (dropArea_counts_us >= areaThresh_counts_us);

    // Beam broken if depth, debounce, and area satisfied
    bool brokenAnalog = (drop >= gDropThreshold) && debounceOK && areaOK;

    // If DO is enabled, treat stable LOW as broken as well
    bool brokenDO = false;
    if (gUseDOTrigger) {
      brokenDO = (lastDO == LOW) && ((nowMs - tLastDOChange) >= (unsigned long)DO_DEBOUNCE_MS);
    }

    bool broken = brokenAnalog || brokenDO;

    // Beam intact ("ON") only after ARM ms near baseline
    bool intact = (tIdleStartMs != 0) && ((nowMs - tIdleStartMs) >= (unsigned long)gIdleArmMs);

    int newState = gLastState;
    if (broken) {
      newState = 0; // OFF (broken)
    } else if (intact) {
      newState = 1; // ON  (intact)
    }
    // If neither broken nor fully intact, keep last state (prevents chatter)

    if (newState != gLastState && newState != -1) {
      gLastState = newState;
      if (gLastState == 1) {
        Serial.println("state:ON");
      } else {
        Serial.println("state:OFF");
      }
      blipLed(); // small visual on any state change
    }

  } else { // ---------- OUTPUT_ZERO or OUTPUT_LEGACY_T ----------
    bool inRefractory = (millis() - tLastTriggerMs) < (unsigned long)gRefractoryMs;
    bool armed        = (tIdleStartMs != 0) && ((millis() - tIdleStartMs) >= (unsigned long)gIdleArmMs);

    if (gAnalogTriggerEn && !inRefractory && armed) {
      if (drop >= gDropThreshold) {
        if (tDipStartMs == 0) tDipStartMs = nowMs;

        bool debounceOK = ((nowMs - tDipStartMs) >= (unsigned long)gDebounceMs);
        long areaThresh_counts_us = (long)gMinAreaCountMs * 1000L;
        bool areaOK = (dropArea_counts_us >= areaThresh_counts_us);

        if (debounceOK && areaOK) {
          emitBrokenToken();      // <-- '0' (default) or 'T' (legacy)
          // re-arm only after baseline again
          tIdleStartMs = 0;
          tDipStartMs  = 0;
          dropArea_counts_us = 0;
        }
      } else {
        tDipStartMs = 0;
      }
    } else {
      // Track dip timer even if not armed/refractory; won't emit
      if (drop >= gDropThreshold) { if (tDipStartMs == 0) tDipStartMs = nowMs; }
      else tDipStartMs = 0;
    }

    // Digital DO trigger (optional)
    if (gUseDOTrigger) {
      if (doLevel != lastDO) { lastDO = doLevel; tLastDOChange = nowMs; }
      if (lastDO == LOW && (nowMs - tLastDOChange) >= (unsigned long)DO_DEBOUNCE_MS) {
        if (!inRefractory) {
          emitBrokenToken();      // <-- '0' (default) or 'T' (legacy)
          tIdleStartMs = 0;  // require idle before next
          tDipStartMs  = 0;
          dropArea_counts_us = 0;
        }
      }
    }
  }

  // Auto-off for LED pulse (does not affect MARK START/END)
  if (tLedPulseStartMs != 0 && (millis() - tLedPulseStartMs) >= (unsigned long)gLedPulseMs) {
    digitalWrite(PIN_SYNC, LOW);
    tLedPulseStartMs = 0;
  }
}
