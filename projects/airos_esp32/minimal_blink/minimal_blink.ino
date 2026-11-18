/* Absolute minimum ESP32 blink.
   No libraries, no Serial, no version-dependent API calls.
   If this fails to upload, the problem is not the code.

   LED_PIN: GPIO 2 is the built-in BLUE led on most ESP32 dev boards
   (DOIT DEVKIT V1, NodeMCU-32S and the usual clones).

   NOTE ON THE RED LED: on nearly every ESP32 dev board the red LED is a
   POWER indicator wired straight to 3.3V through a resistor. It is not
   connected to any GPIO and cannot be blinked by software — it is on
   whenever the board has power, and that is all it ever does.
   The controllable one is the blue LED, normally GPIO 2.

   If GPIO 2 does nothing, try 5, then 22, then 16. Boards vary.       */

#define LED_PIN 2

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
    delay(500);
}
