#include <WiFi.h>

// Motor Pins
#define m1 21  // Right Motor MA1
#define m2 19  // Right Motor MA2
#define m3 18  // Left Motor MB1
#define m4 5   // Left Motor MB2

// Enable Pins
#define e1 15  // Right Motor Enable Pin EA
#define e2 2   // Left Motor Enable Pin EB

// Infrared Sensor Pins
#define ir1 32
#define ir2 33
#define ir3 25
#define ir4 26
#define ir5 27

#define GL 12
#define RL 4
#define buzz 14

#define mspeed 255
#define tspeed 200

int s1;
int s2;
int s3;
int s4;
int s5;

int currentInstructionIndex = 0;
int flag = 0;
unsigned long delayStartTime = 0;

// WiFi credentials
const char* ssid = "Aman";                    // Enter your wifi hotspot ssid
const char* password =  "1234567890";              // Enter your wifi hotspot password
const uint16_t port = 8002;
const char * host = "192.168.189.139";             // Enter the IP address of your laptop after connecting it to wifi hotspot

char incomingPacket[80];
WiFiClient client;

String msg;
bool gotString = false;  // Assuming 'gotString' is a boolean variable
char instructions[80];

void setup() {

  Serial.begin(9600);
  // Motor setup
  pinMode(m1, OUTPUT);
  pinMode(m2, OUTPUT);
  pinMode(m3, OUTPUT);
  pinMode(m4, OUTPUT);
  pinMode(e1, OUTPUT);
  pinMode(e2, OUTPUT);
  pinMode(GL, OUTPUT);
  pinMode(RL, OUTPUT);

  pinMode(buzz, OUTPUT);
  // Infrared Sensor setup
  pinMode(ir1, INPUT);
  pinMode(ir2, INPUT);
  pinMode(ir3, INPUT);
  pinMode(ir4, INPUT);
  pinMode(ir5, INPUT);

  digitalWrite(buzz, HIGH);
  // Connecting to wifi
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
  }

  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (!gotString) {
    if (!client.connect(host, port)) {
      Serial.println("Connection to host failed");
      delay(200);
      return;
    }
    client.print("Hello from GG#1126!");
    // Move the declaration of 'instructions' outside the loop
    while (1) {
      msg = client.readStringUntil('\n');
      msg.trim();
      Serial.println(msg);

      // Convert the String to a character array
      msg.toCharArray(instructions, msg.length() + 1);

      // Display the character array
      Serial.print("Instruction Array: ");
      Serial.println(instructions);
      gotString = true;
      // Process the instruction array as needed
      break;
    }
    Serial.println(instructions);
    delay(5000);
  }

  if (flag == 0) {
    startleap();
    flag++;
  }
  readline();
  if ((s1 == 0) && (s2 == 1) && (s3 == 1) && (s4 == 1) && (s5 == 0) ||
      (s1 == 1) && (s2 == 1) && (s3 == 1) && (s4 == 1) && (s5 == 0) ||
      (s1 == 1) && (s2 == 1) && (s3 == 1) && (s4 == 0) && (s5 == 0) ||
      (s1 == 0) && (s2 == 1) && (s3 == 1) && (s4 == 1) && (s5 == 1) ||
      (s1 == 0) && (s2 == 0) && (s3 == 1) && (s4 == 1) && (s5 == 1) ||
      (s1 == 1) && (s2 == 1) && (s3 == 1) && (s4 == 1) && (s5 == 1)) {

    straight();
    delay(320);
    unsigned long currentTime = millis();
    unsigned long delayTime = 70; // 400 ms delay
    while (millis() - currentTime < delayTime) {
      follow_line();
      avoid_line();
    }
    stopmotor();
    Serial.print("Instruction Array: ");
    Serial.println(instructions);

    unsigned long currentTime_halt = millis();
    char instruction = instructions[currentInstructionIndex];
    if (instructions[currentInstructionIndex + 1] == 'O') {
      executeInstruction(instruction);
      delay(30);
      while (millis() - currentTime_halt < 2450) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
    }

    else if (instructions[currentInstructionIndex + 1] == 'U') {
      executeInstruction(instruction);
      delay(30);
      while (millis() - currentTime_halt < 2900) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
    }

    else if (instructions[currentInstructionIndex + 1] == 'V') {
      executeInstruction(instruction);
      delay(30);
      while (millis() - currentTime_halt < 3200) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
    }

    else if (instructions[currentInstructionIndex + 1] == 'X') {
      executeInstruction(instruction);
      delay(30);
      while (millis() - currentTime_halt < 4150) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
    }

    else if (instructions[currentInstructionIndex + 1] == 'W') {
      executeInstruction(instruction);
      delay(30);
      while (millis() - currentTime_halt < 4050) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
    }

    else {
      executeInstruction(instruction);
      currentInstructionIndex++;
      delay(30);
      if (currentInstructionIndex == strlen(instructions)) {
        unsigned long currentTime = millis();
        unsigned long delayTimes = 1100; // 400 ms delay
        while (millis() - currentTime < delayTimes) {
          follow_line();
        }
        digitalWrite(m1, LOW);
        digitalWrite(m2, LOW);
        digitalWrite(m3, LOW);
        digitalWrite(m4, LOW);
        analogWrite(e1, 0);
        analogWrite(e2, 0);
        digitalWrite(RL, HIGH);
        lbeep();
        digitalWrite(RL, LOW);
      }
    }
  }
  avoid_line();
  follow_line();
}

void readline() {
  // Infrared sensor readings
  s1 = digitalRead(ir5);  // Left Most Sensor
  s2 = digitalRead(ir4);  // Left Sensor
  s3 = digitalRead(ir3);  // Middle Sensor
  s4 = digitalRead(ir2);  // Right Sensor
  s5 = digitalRead(ir1);  // Right Most Sensor
}

// Motor movement functions

void startleap() {
  digitalWrite(m1, HIGH);
  digitalWrite(m2, LOW);
  digitalWrite(m3, HIGH);
  digitalWrite(m4, LOW);
  analogWrite(e1, mspeed);
  analogWrite(e2, mspeed);
  delay(50);
}

void turnleft() {
  readline();
  digitalWrite(m1, LOW);
  digitalWrite(m2, HIGH);
  digitalWrite(m3, HIGH);
  digitalWrite(m4, LOW);
  analogWrite(e1, 200);
  analogWrite(e2, 200);
  delay(250);
  while (digitalRead(ir3) != 1) {
    digitalWrite(m1, LOW);
    digitalWrite(m2, HIGH);
    digitalWrite(m3, HIGH);
    digitalWrite(m4, LOW);
    analogWrite(e1, tspeed);
    analogWrite(e2, tspeed);
  }
}

void correctleft() {
  digitalWrite(m1, LOW);
  digitalWrite(m2, HIGH);
  digitalWrite(m3, HIGH);
  digitalWrite(m4, LOW);
  analogWrite(e1, mspeed);
  analogWrite(e2, mspeed);
}

void turnright() {
  readline();
  digitalWrite(m1, HIGH);
  digitalWrite(m2, LOW);
  digitalWrite(m3, LOW);
  digitalWrite(m4, HIGH);
  analogWrite(e1, tspeed);
  analogWrite(e2, tspeed);
  delay(350);
  while (digitalRead(ir3) != 1) {
    digitalWrite(m1, HIGH);
    digitalWrite(m2, LOW);
    digitalWrite(m3, LOW);
    digitalWrite(m4, HIGH);
    analogWrite(e1, tspeed);
    analogWrite(e2, tspeed);
  }
  readline();
}

void correctright() {
  digitalWrite(m1, HIGH);
  digitalWrite(m2, LOW);
  digitalWrite(m3, LOW);
  digitalWrite(m4, HIGH);
  analogWrite(e1, mspeed);
  analogWrite(e2, mspeed);
  readline();
}

void straight() {
  digitalWrite(m1, HIGH);
  digitalWrite(m2, LOW);
  digitalWrite(m3, HIGH);
  digitalWrite(m4, LOW);
  analogWrite(e1, mspeed);
  analogWrite(e2, mspeed);
  readline();
}

void stopmotor() {
  digitalWrite(m1, LOW);
  digitalWrite(m2, LOW);
  digitalWrite(m3, LOW);
  digitalWrite(m4, LOW);
  analogWrite(e1, 0);
  analogWrite(e2, 0);
  digitalWrite(RL, HIGH);
  beep();
}

// Execute given movement instruction
void executeInstruction(char instruction) {
  switch (instruction) {
    case 'F':
      straight();
      break;
    case 'L':
      turnleft();
      break;
    case 'R':
      turnright();
      break;
  }
  Serial.println(instruction);
}

void follow_line() {

  readline();
  if ((s1 == 0) && (s2 == 0) && (s3 == 0) && (s4 == 0) && (s5 == 0) ||
      (s1 == 0) && (s2 == 0) && (s3 == 1) && (s4 == 0) && (s5 == 0)) {
    straight();
  }

  if ((s1 == 0) && (s2 == 1) && (s3 == 1) && (s4 == 0) && (s5 == 0) ||
      (s1 == 0) && (s2 == 1) && (s3 == 0) && (s4 == 0) && (s5 == 0)) {
    correctright();
  }

  if ((s1 == 0) && (s2 == 0) && (s3 == 1) && (s4 == 1) && (s5 == 0) ||
      (s1 == 0) && (s2 == 0) && (s3 == 0) && (s4 == 1) && (s5 == 0)) {
    correctleft();
  }
}

void avoid_line() {
  if ((s1 == 0) && (s2 == 0) && (s3 == 0) && (s4 == 0) && (s5 == 1)) {
    correctright();
  }

  if ((s1 == 1) && (s2 == 0) && (s3 == 0) && (s4 == 0) && (s5 == 0)) {
    correctleft();
  }

  if ((s1 == 1) && (s2 == 1) && (s3 == 1) && (s4 == 1) && (s5 == 1)) {
    straight();
  }
}
void beep() {
  digitalWrite(buzz, LOW);
  delay(1000);
  digitalWrite(buzz, HIGH);
  delay(500);
}

void lbeep() {
  digitalWrite(buzz, LOW);
  delay(5000);
  digitalWrite(buzz, HIGH);
  delay(250);
}
