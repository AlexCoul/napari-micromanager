#include <RGBmatrixPanel.h>

// Most of the signal pins are configurable, but the CLK pin has some
// special constraints.  On 8-bit AVR boards it must be on PORTB...
// Pin 11 works on the Arduino Mega.  On 32-bit SAMD boards it must be
// on the same PORT as the RGB data pins (D2-D7)...
// Pin 8 works on the Adafruit Metro M0 or Arduino Zero,
// Pin A4 works on the Adafruit Metro M4 (if using the Adafruit RGB
// Matrix Shield, cut trace between CLK pads and run a wire to A4).

#define CLK  8   // USE THIS ON ADAFRUIT METRO M0, etc.
//#define CLK A4 // USE THIS ON METRO M4 (not M0)
//#define CLK 11 // USE THIS ON ARDUINO MEGA
#define OE   9
#define LAT 10
#define A   A0
#define B   A1
#define C   A2
#define D   A3

// epifluo RED LED
// only pins 12 and 13 left by the LED array
// see https://learn.adafruit.com/32x16-32x32-rgb-led-matrix/connecting-with-jumper-wires
#define EPI_LED 13

// number of inner LEDs that are OFF
#define NB_BLACK 1

RGBmatrixPanel matrix(A, B, C, D, CLK, LAT, OE, false);

// parameters for reading strings from serial port
String inputString = "";


void setup() {
  // put your setup code here, to run once:
  //setup serial port
  Serial.begin(115200);
  inputString.reserve(200);

  //setup epifluo LED pin
  pinMode(EPI_LED, OUTPUT);
  
  // setup LED array
  matrix.begin();
  matrix.fillRect(0, 0, 32, 32, matrix.Color333(0, 0, 0));

}
 
void loop() {
    if(Serial.available()){
        inputString = Serial.readStringUntil('\n');
          
        if (inputString == "ALIGN"){
          Serial.println("starting align");
          digitalWrite(EPI_LED, LOW);
          matrix.fillScreen(matrix.Color333(7, 7, 7));
          matrix.fillRect(16-NB_BLACK, 16-NB_BLACK, 2*NB_BLACK, 2*NB_BLACK, matrix.Color333(0, 0, 0));
        }
        else if (inputString == "FULL"){
          Serial.println("starting full");
          digitalWrite(EPI_LED, LOW);
          matrix.fillScreen(matrix.Color333(7, 7, 7));
        }
        else if (inputString == "DPC0"){
          Serial.println("DPC0");
          digitalWrite(EPI_LED, LOW);
          matrix.fillRect(0, 0, 32, 16-NB_BLACK, matrix.Color333(7, 7, 7));
          matrix.fillRect(0, 16-NB_BLACK, 32, 16+NB_BLACK, matrix.Color333(0, 0, 0));
        }
        else if (inputString == "DPC1"){
          Serial.println("DPC1");
          digitalWrite(EPI_LED, LOW);
          matrix.fillRect(0, 0, 32, 16+NB_BLACK, matrix.Color333(0, 0, 0));
          matrix.fillRect(0, 16+NB_BLACK, 32, 16-NB_BLACK, matrix.Color333(7, 7, 7));
        }
        else if (inputString == "DPC2"){
          Serial.println("DPC2");
          digitalWrite(EPI_LED, LOW);
          matrix.fillRect(0, 0, 16-NB_BLACK, 32, matrix.Color333(7, 7, 7));
          matrix.fillRect(16-NB_BLACK, 0, 16+NB_BLACK, 32, matrix.Color333(0, 0, 0));
        }
        else if (inputString == "DPC3"){
          Serial.println("DPC3");
          digitalWrite(EPI_LED, LOW);
          matrix.fillRect(0, 0, 16+NB_BLACK, 32, matrix.Color333(0, 0, 0));
          matrix.fillRect(16+NB_BLACK, 0, 16-NB_BLACK, 32, matrix.Color333(7, 7, 7));
        }
        else if (inputString == "LED"){
          Serial.println("LED");
          matrix.fillScreen(matrix.Color333(0, 0, 0));
          digitalWrite(EPI_LED, HIGH);
        }
        else if (inputString == "OFF"){
          Serial.println("OFF");
          digitalWrite(EPI_LED, LOW);
          matrix.fillScreen(matrix.Color333(0, 0, 0));
        }
        else if (inputString == "CROSS"){
          // make a cross, useful to center LED array on coverslip
          Serial.println("CROSS");
          digitalWrite(EPI_LED, LOW);
          matrix.fillScreen(matrix.Color333(0, 0, 0));
          matrix.fillRect(0, 16-1, 32, 2, matrix.Color333(7, 7, 7));
          matrix.fillRect(16-1, 0, 2, 32, matrix.Color333(7, 7, 7));
        }
        else {
          Serial.println("Not recognized: " + inputString);
        }
    }
}
