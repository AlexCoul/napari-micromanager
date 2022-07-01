#include <Adafruit_NeoPixel.h>
#include <SoftwareSerial.h>

// input trigger line
#define INPUT_PIN 7

// set up NeoPixel LEDs
// digital line NeoPixel data in line is connected to
#define LED_PIN 6
// number of LED's
#define LED_COUNT 24
Adafruit_NeoPixel strip(LED_COUNT, LED_PIN);

// parameters for reading strings from serial port
String inputString = "";
boolean stringComplete = false;

// other parameters
boolean seq_running = false;
boolean start_align = false;
boolean stop_align = false;
boolean cycle_leds = false;
boolean half_leds = false;
boolean align_leds = false;
boolean all_leds = false;
boolean off = true;
boolean dpc0 = false;
boolean dpc1 = false;
boolean dpc2 = false;
boolean dpc3 = false;
boolean led = false;


void setup() {
  // put your setup code here, to run once:
  //setup serial port
  Serial.begin(115200);
  inputString.reserve(200);

  //setup pins
  pinMode(13, OUTPUT);
  pinMode(2, OUTPUT);
  pinMode(INPUT_PIN, INPUT);
  
  // setup NeoPixel
  strip.begin();
  strip.show();

}

void loop() {
  // wait for serial command starting sequence
  if(stringComplete) {

    off = true;
    if(off) {
      for (int n=0;n<LED_COUNT;n++) {
        strip.setPixelColor(n, 0, 0, 0);
      }
      strip.show();
      digitalWrite(2,LOW);
      off = false;
    }

    if (inputString == "ALIGN\n"){
      off = false;
      start_align = true;
      Serial.println("starting align");
    }
    else if (inputString == "STOP ALIGN\n"){
      off = false;
      stop_align = true;
      Serial.println("stopping align");
    }
    else if (inputString == "ALIGN 4\n") {
      off = false;
      align_leds = true;
      Serial.println("starting corner align");
    }
    else if (inputString == "DPC0\n"){
      off = false;
      dpc0 = true;
      Serial.println("DPC0");
    }
    else if (inputString == "DPC1\n"){
      off = false;
      dpc1 = true;
      Serial.println("DPC1");
    }
    else if (inputString == "DPC2\n"){
      off = false;
      dpc2 = true;
      Serial.println("DPC2");
    }
    else if (inputString == "DPC3\n"){
      off = false;
      dpc3 = true;
      Serial.println("DPC3");
    }
    else if (inputString == "LED\n"){
      off = false;
      led = true;
      Serial.println("LED");
    }
    else if (inputString == "OFF\n"){
      off = true;
      Serial.println("OFF");
    }

  else {
    Serial.println(inputString);
  }
    
    // clear the string:
    inputString = "";
    stringComplete = false;
  }
  
  // turn on alignment LEDs
  if(start_align){
    for (int n=0;n<LED_COUNT;n++) {
      strip.setPixelColor(n, 0, 100, 0);
    }
    strip.show();
    start_align = false;
  }

    
  // turn off alignment LEDs
  if(stop_align){
    for (int n=0;n<LED_COUNT;n++) {
      strip.setPixelColor(n, 0, 0, 0);
    }
    strip.show();
    stop_align = false;
  }

   // turn on half of LEDS
   if(dpc0){
      for (int n=0;n<LED_COUNT/2;n++) {
        strip.setPixelColor(n, 255, 255, 255);
      }
      for (int n=LED_COUNT/2;n<LED_COUNT;n++) {
        strip.setPixelColor(n, 0, 0, 0);
      }
      strip.show();
      dpc0 = false;
    }

    if(dpc1){
      for (int n=0;n<LED_COUNT/2;n++) {
        strip.setPixelColor(n, 0, 0, 0);
      }
      for (int n=LED_COUNT/2;n<LED_COUNT;n++) {
        strip.setPixelColor(n, 255, 255, 255);
      }
      strip.show();
      dpc1 = false;
    }

    if(dpc2){
      for (int n=LED_COUNT/4;n<3*LED_COUNT/4;n++) {
        strip.setPixelColor(n, 255, 255, 255);
      }
      for (int n=0;n<LED_COUNT/4;n++) {
        strip.setPixelColor(n, 0, 0, 0);
      }
      for (int n=3*LED_COUNT/4;n<LED_COUNT;n++) {
        strip.setPixelColor(n, 0, 0, 0);
      }
      strip.show();
      dpc2 = false;
    }

    if(dpc3){
      for (int n=LED_COUNT/4;n<3*LED_COUNT/4;n++) {
        strip.setPixelColor(n, 0, 0, 0);
      }
      for (int n=0;n<LED_COUNT/4;n++) {
        strip.setPixelColor(n, 255, 255, 255);
      }
      for (int n=3*LED_COUNT/4;n<LED_COUNT;n++) {
        strip.setPixelColor(n, 255, 255, 255);
      }
      strip.show();
      dpc3 = false;
    }

    if (led) {
      digitalWrite(2,HIGH);
      led = false;
    }

    if(off) {
      for (int n=0;n<LED_COUNT;n++) {
        strip.setPixelColor(n, 0, 0, 0);
      }
      strip.show();
      digitalWrite(2,LOW);
      off = false;
    }
   
 // Turn on 4 corners
   if(align_leds){
      strip.setPixelColor(3, 0, 250, 0);
      strip.setPixelColor(LED_COUNT/4+3, 0, 250, 0);
      strip.setPixelColor(LED_COUNT/2+3, 0, 250, 0);
      strip.setPixelColor(LED_COUNT-LED_COUNT/4+3, 0, 250, 0);
      strip.show();
      align_leds = false;
    }

  
}

// serialEvent() function and related code taken from https://www.arduino.cc/en/Tutorial/SerialEvent
/*
  SerialEvent occurs whenever a new data comes in the hardware serial RX. This
  routine is run between each time loop() runs, so using delay inside loop can
  delay response. Multiple bytes of data may be available.
*/
void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
