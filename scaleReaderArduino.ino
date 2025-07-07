/* --------------------------------------------------------
   HX711 raw-stream firmware  –  v1.0
   Outputs: <millis>\t<raw_counts>\n      @ 20 Hz
   -------------------------------------------------------- */

#include <HX711.h>

ew//pinout on scale driver
//SCK yellow driver side /blue arduino side
//DT orange driver side /green arduin o side
//VCC white
//GND black

// Pin mapping – adjust to your wiring
constexpr byte PIN_DOUT = 2;
constexpr byte PIN_SCK  = 3;

HX711 scale;

// -------- parameters you may tune -------------------------
constexpr unsigned long SAMPLE_PERIOD_MS = 50;   // 20 Hz
constexpr byte   SOFT_AVG = 8;   // =1 → no averaging, >1 → avg N reads
// ----------------------------------------------------------

void setup()
{
    Serial.begin(115200);
    while (!Serial) ;          // wait for host

    scale.begin(PIN_DOUT, PIN_SCK);
    // No tare or calibration here – host does it.
}

void loop()
{
    static unsigned long t_last = 0;
    unsigned long now = millis();
    if (now - t_last >= SAMPLE_PERIOD_MS && scale.is_ready())
    {
        t_last = now;

        long acc = 0;
        for (byte i = 0; i < SOFT_AVG; ++i)
            acc += scale.read();
        long raw = acc / SOFT_AVG;

        Serial.print(now);     // time-stamp first
        Serial.print('\t');
        Serial.println(raw);   // raw 24-bit counts
    }
}
