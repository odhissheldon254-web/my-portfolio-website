/* ─────────────────────────────────────────────────────────────────────────
   AIROS — ESP32 bring-up / step 0
   Board verification before any peripheral is wired.

   This is deliberately more than a blink. It proves four separate things,
   so that when something fails later you know which layer broke:

     1. UPLOAD works            — the sketch is running at all
     2. USB SERIAL works        — you can read output (needed for all
                                  future debugging; without this you are
                                  blind for the rest of the project)
     3. GPIO OUTPUT works       — the LED toggles
     4. TIMING is accurate      — measured loop period vs the 1000 ms it
                                  was asked for. AIROS is a scheduler; if
                                  the clock lies, nothing above it is
                                  trustworthy. This is the number that
                                  actually matters going forward.

   Expected serial output at 115200 baud:

     === AIROS ESP32 bring-up ===
     Chip model    : ESP32-D0WD-V3 rev 3
     CPU cores     : 2
     ...
     beat 1  | uptime 1002 ms | period 1001 ms | drift +1 ms
   ───────────────────────────────────────────────────────────────────── */

/* Onboard LED. GPIO 2 on DOIT DEVKIT V1 and most 30-pin clones.
   If nothing blinks but serial works, your board either has no onboard
   LED or wires it elsewhere — try 5, or 22. The serial output below is
   the real proof of life, so a dark LED is not a failure. */
#define LED_PIN 2

/* How long each cycle should take, in milliseconds. */
#define CYCLE_MS 1000
#define ON_MS     150

static uint32_t beat      = 0;
static uint32_t last_edge = 0;

void setup() {
    Serial.begin(115200);
    delay(600);                    /* let USB CDC / the adapter settle */

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.println();
    Serial.println(F("=== AIROS ESP32 bring-up ==="));
    Serial.printf("Chip model    : %s rev %d\n",
                  ESP.getChipModel(), ESP.getChipRevision());
    Serial.printf("CPU cores     : %d\n",   ESP.getChipCores());
    Serial.printf("CPU frequency : %u MHz\n", (unsigned)getCpuFrequencyMhz());
    Serial.printf("Flash size    : %u bytes\n", (unsigned)ESP.getFlashChipSize());
    Serial.printf("Free heap     : %u bytes\n", (unsigned)ESP.getFreeHeap());
    Serial.printf("SDK version   : %s\n",   ESP.getSdkVersion());

    uint64_t mac = ESP.getEfuseMac();
    Serial.printf("MAC (efuse)   : %04X%08X\n",
                  (uint16_t)(mac >> 32), (uint32_t)mac);

    Serial.println();
    Serial.println(F("If you can read this line, upload AND serial both work."));
    Serial.printf("LED on GPIO %d should blink once per second.\n", LED_PIN);
    Serial.println(F("Watch the drift column: it should stay within a few ms."));
    Serial.println();

    last_edge = millis();
}

void loop() {
    uint32_t cycle_start = millis();

    digitalWrite(LED_PIN, HIGH);
    delay(ON_MS);
    digitalWrite(LED_PIN, LOW);
    delay(CYCLE_MS - ON_MS);

    uint32_t now    = millis();
    uint32_t period = now - last_edge;
    last_edge       = now;
    beat++;

    /* Signed drift against the requested period. On a healthy ESP32 this
       sits within a couple of milliseconds. Consistent large drift would
       mean the tick source cannot be trusted for real-time scheduling. */
    int32_t drift = (int32_t)period - (int32_t)CYCLE_MS;

    Serial.printf("beat %-4u | uptime %7u ms | period %4u ms | drift %+d ms\n",
                  (unsigned)beat, (unsigned)now, (unsigned)period, (int)drift);

    (void)cycle_start;
}
