// Flash at 115200 baud
// ===== ElegooFlyPySync.ino =====
// Elegoo UNO R3 (CH340). Sync FlyPy via USB serial.
// D2: trigger-in (button to GND, INPUT_PULLUP).
// D8: TTL sync pulses (also drives in-frame LED).
// D9: LIGHT control (LED or simple NPN driver).

const uint8_t PIN_TRIGGER_IN = 2;
const uint8_t PIN_TTL_OUT    = 8;
const uint8_t PIN_LIGHT_OUT  = 9;

const uint16_t DEFAULT_PULSE_MS = 20;
const uint16_t DEBOUNCE_MS = 30;

volatile bool trigFlag = false;
volatile uint32_t lastEdgeMs = 0;

void onTrigISR() {
  uint32_t now = millis();
  if (now - lastEdgeMs >= DEBOUNCE_MS) {
    lastEdgeMs = now;
    trigFlag = true;
  }
}

void pulse(uint16_t ms = DEFAULT_PULSE_MS) {
  digitalWrite(PIN_TTL_OUT, HIGH);
  delay(ms);
  digitalWrite(PIN_TTL_OUT, LOW);
}

void lightsOn()  { digitalWrite(PIN_LIGHT_OUT, HIGH); }
void lightsOff() { digitalWrite(PIN_LIGHT_OUT, LOW);  }

String line;

void setup() {
  pinMode(PIN_TRIGGER_IN, INPUT_PULLUP);
  pinMode(PIN_TTL_OUT, OUTPUT);
  pinMode(PIN_LIGHT_OUT, OUTPUT);
  digitalWrite(PIN_TTL_OUT, LOW);
  digitalWrite(PIN_LIGHT_OUT, LOW);

  Serial.begin(115200);
  delay(50);

  attachInterrupt(digitalPinToInterrupt(PIN_TRIGGER_IN), onTrigISR, FALLING);
  Serial.println("ELEGOO READY");
}

void loop() {
  // Forward external trigger to FlyPy
  if (trigFlag) {
    trigFlag = false;
    Serial.println("T");  // FlyPy treats this as a trigger
  }

  // Parse one command line from FlyPy (newline-terminated)
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c != '\n') { line += c; continue; }

    line.trim();
    if (line.length() > 0) {
      String cmd = line; cmd.toUpperCase();

      if (cmd == "START" || cmd == "STIM" || cmd == "END") {
        pulse(DEFAULT_PULSE_MS);
        Serial.println("ACK " + line);
      } else if (cmd.startsWith("PULSE")) {
        int sp = cmd.indexOf(' ');
        uint16_t ms = (sp > 0) ? (uint16_t)cmd.substring(sp + 1).toInt() : DEFAULT_PULSE_MS;
        if (ms == 0) ms = DEFAULT_PULSE_MS;
        pulse(ms);
        Serial.print("ACK PULSE "); Serial.println(ms);
      } else if (cmd == "LIGHT ON") {
        lightsOn();  Serial.println("ACK LIGHT ON");
      } else if (cmd == "LIGHT OFF") {
        lightsOff(); Serial.println("ACK LIGHT OFF");
      } else if (cmd.startsWith("MARK")) {
        digitalWrite(PIN_TTL_OUT, HIGH); delay(5); digitalWrite(PIN_TTL_OUT, LOW);
        Serial.println("ACK " + line);
      } else {
        Serial.println("ERR UNKNOWN " + line);
      }
    }
    line = "";
  }
}
