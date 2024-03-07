/*
 * Team Id: 1126
 * Author List: Vishakha Fulare, Aman Raut, Viranchi Dakhare, Gagan Loya
 * Filename: GG_task_6_C_code.ino
 * Theme: Geo Guide
 * Functions:
   - setup()
   - loop()
   - readline()
   - startleap()
   - turnleft()
   - correctleft()
   - turnright()
   - correctright()
   - straight()
   - stopmotor()
   - executeInstruction(char instruction)
   - follow_line()
   - avoid_line()
   - beep()
   - lbeep()
 * Global Variables:
   - int s1;  // State of the leftmost infrared sensor
   - int s2;  // State of the left infrared sensor
   - int s3;  // State of the middle infrared sensor
   - int s4;  // State of the right infrared sensor
   - int s5;  // State of the rightmost infrared sensor

   - int currentInstructionIndex = 0;  // Index of the current instruction in the instructions array
   - int flag = 0;  // Flag used to control the execution of certain parts of the code

   - unsigned long delayStartTime = 0;  // Records the start time for specific delays

   - const char* ssid = "Aman";  // WiFi hotspot SSID (Wi-Fi network name)
   - const char* password = "1234567890";  // WiFi hotspot password
   - const uint16_t port = 8002;  // Port number used for communication with the host
   - const char* host = "192.168.70.139";  // IP address of the host (laptop) to which the Arduino connects

   - char incomingPacket[80];  // Incoming packet data from the host
   - WiFiClient client;  // WiFi client for communication with the host
   - String msg;  // Received message from the host

   - bool gotString = false;  // Indicates whether a complete string has been received from the host
   - char instructions[80];  // Processed instructions received from the host
*/

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
const char * host = "192.168.70.139";             // Enter the IP address of your laptop after connecting it to wifi hotspot

char incomingPacket[80];
WiFiClient client;

String msg;
bool gotString = false;  // Assuming 'gotString' is a boolean variable
char instructions[80];

/*
 * Function Name: setup()
 * Input: None
 * Output: None
 * Logic: The setup function initializes the Arduino board, sets up communication, and configures pins for motors and sensors.
 *        It also connects to WiFi and prints the assigned IP address to the serial monitor.
 * Example Call: Automatically called by the Arduino framework at the beginning of the program.
 */

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


/*
 * Function Name: loop()
 * Input: None
 * Output: None
 * Logic: The main loop of the program responsible for controlling the bot's behavior based on received instructions.
 *        It establishes a connection to the host and sends a greeting message.
 *        Upon receiving instructions, it processes the instruction array and executes corresponding actions:
 *        - Movement commands (F, L, R) are executed with appropriate delays and line following/avoidance.
 *        - Special commands (O, U, V, X, W) trigger specific actions with different delay durations.
 *        - The loop continues until all instructions are processed, ending with a stopmotor() and lbeep() sequence.
 * Example Call: Automatically called by the Arduino framework.
 */

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
    delay(3000);
    digitalWrite(RL,HIGH);
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
    delay(300);
    unsigned long currentTime = millis();
    unsigned long delayTime = 70; // 400 ms delay
    while (millis() - currentTime < delayTime) {
      follow_line();
      avoid_line();
    }
    //stopmotor();
    Serial.print("Instruction Array: ");
    Serial.println(instructions);

    unsigned long currentTime_halt = millis();
    char instruction = instructions[currentInstructionIndex];
    if (instructions[currentInstructionIndex + 1] == 'O') { //D
      executeInstruction(instruction);
      delay(30);
      while (millis() - currentTime_halt < 2200) {
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
      if((instructions[currentInstructionIndex] == 'R')||(instructions[currentInstructionIndex] == 'L')){
         while (millis() - currentTime_halt < 2700) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
      }
      else{
      while (millis() - currentTime_halt < 2800) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
      }
    }

    else if (instructions[currentInstructionIndex + 1] == 'V') {
      executeInstruction(instruction);
      delay(30);
      if((instructions[currentInstructionIndex] == 'R')||(instructions[currentInstructionIndex] == 'L')){
         while (millis() - currentTime_halt < 3250) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
      }
      else{
      while (millis() - currentTime_halt < 2600) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();}
    }

    else if (instructions[currentInstructionIndex + 1] == 'X') { //X
      executeInstruction(instruction);
      delay(30);
      while (millis() - currentTime_halt < 3400) { //9000 FOR L 
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
      }

    else if (instructions[currentInstructionIndex + 1] == 'W') { //A
      executeInstruction(instruction);
      delay(30);
      if((instructions[currentInstructionIndex] == 'R')||(instructions[currentInstructionIndex] == 'L')){
         while (millis() - currentTime_halt < 3500) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
      }
      else{
      while (millis() - currentTime_halt < 3100) {
        follow_line();
        avoid_line();
        readline();
      }
      currentInstructionIndex = currentInstructionIndex + 2;
      stopmotor();
      startleap();
      }
    }

    else {
      executeInstruction(instruction);
      currentInstructionIndex++;
      delay(30);
      if (currentInstructionIndex == strlen(instructions)) {
        unsigned long currentTime = millis();
        unsigned long delayTimes = 1500; // 400 ms delay
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
  follow_line();
  avoid_line();
}

/*
 * Function Name: readline()
 * Input: None
 * Output: None
 * Logic: This function reads the values from infrared sensors and updates corresponding variables.
 *        It reads the digital signals from five infrared sensors (ir1 to ir5) and assigns the values to global variables (s1 to s5):
 *        - s1: Leftmost sensor
 *        - s2: Left sensor
 *        - s3: Middle sensor
 *        - s4: Right sensor
 *        - s5: Rightmost sensor
 * Example Call: readline();
 */

void readline() {
  // Infrared sensor readings
  s1 = digitalRead(ir5);  // Left Most Sensor
  s2 = digitalRead(ir4);  // Left Sensor
  s3 = digitalRead(ir3);  // Middle Sensor
  s4 = digitalRead(ir2);  // Right Sensor
  s5 = digitalRead(ir1);  // Right Most Sensor
}


// Motor movement functions
/*
 * Function Name: startleap()
 * Input: None
 * Output: None
 * Logic: This function initiates a leap motion by activating specific motors.
 *        It activates motor 1 and motor 3, causing the bot to move forward while keeping motor 2 and motor 4 off.
 *        Additionally, it sets the motor speeds using analogWrite() with the predefined speed value (mspeed).
 *        The function introduces a delay of 50 milliseconds to allow the bot to start the leap.
 * Example Call: startleap();
 */
void startleap() {
  digitalWrite(m1, HIGH);
  digitalWrite(m2, LOW);
  digitalWrite(m3, HIGH);
  digitalWrite(m4, LOW);
  analogWrite(e1, mspeed);
  analogWrite(e2, mspeed);
  delay(50);
}

/*
 * Function Name: turnleft()
 * Input: None
 * Output: None
 * Logic: This function commands the bot to turn left by activating specific motors.
 *        It initiates the turn by activating motor 2 and motor 3, causing the bot to turn left while keeping motor 1 and motor 4 off.
 *        Additionally, it sets the motor speeds using analogWrite() with a predefined speed value (200).
 *        It introduces a delay of 250 milliseconds to allow the bot to start the turn.
 *        The function then enters a loop, continuing the left turn until the third infrared sensor (ir3) detects an obstacle.
 *        The loop ensures that the bot maintains the left turn until an obstacle is encountered.
 * Example Call: turnleft();
 */
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

/*
 * Function Name: correctleft()
 * Input: None
 * Output: None
 * Logic: This function corrects the bot's path to the left by activating specific motors.
 *        It activates motor 2 and motor 3, causing the bot to turn to the left while keeping motor 1 and motor 4 off.
 *        Additionally, it sets the motor speeds using analogWrite() with the predefined speed value (mspeed).
 * Example Call: correctleft();
 */
void correctleft() {
  digitalWrite(m1, LOW);
  digitalWrite(m2, HIGH);
  digitalWrite(m3, HIGH);
  digitalWrite(m4, LOW);
  analogWrite(e1, mspeed);
  analogWrite(e2, mspeed);
}

/*
 * Function Name: turnright()
 * Input: None
 * Output: None
 * Logic: This function commands the bot to turn right by activating specific motors.
 *        It initiates the turn by activating motor 1 and motor 4, causing the bot to turn right while keeping motor 2 and motor 3 off.
 *        Additionally, it sets the motor speeds using analogWrite() with the predefined turn speed value (tspeed).
 *        It introduces a delay of 350 milliseconds to allow the bot to start the turn.
 *        The function then enters a loop, continuing the right turn until the third infrared sensor (ir3) detects an obstacle.
 *        The loop ensures that the bot maintains the right turn until an obstacle is encountered.
 *        The function concludes by reading line sensor values using the readline() function.
 * Example Call: turnright();
 */
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

/*
 * Function Name: correctright()
 * Input: None
 * Output: None
 * Logic: This function corrects the bot's path to the right by activating specific motors.
 *        It activates motor 1 and motor 4, causing the bot to turn to the right while keeping motor 2 and motor 3 off.
 *        Additionally, it sets the motor speeds using analogWrite() with the predefined speed value (mspeed).
 *        The function also reads line sensor values using the readline() function.
 * Example Call: correctright();
 */
void correctright() {
  digitalWrite(m1, HIGH);
  digitalWrite(m2, LOW);
  digitalWrite(m3, LOW);
  digitalWrite(m4, HIGH);
  analogWrite(e1, mspeed);
  analogWrite(e2, mspeed);
  readline();
}

/*
 * Function Name: straight()
 * Input: None
 * Output: None
 * Logic: This function commands the bot to move straight by setting the motor control pins.
 *        It activates the motors on one side (m1 and m3) while keeping the motors on the other side (m2 and m4) off.
 *        Additionally, it sets the motor speeds using analogWrite() with the predefined speed value (mspeed).
 *        The function also reads line sensor values using the readline() function.
 * Example Call: straight();
 */
void straight() {
  digitalWrite(m1, HIGH);
  digitalWrite(m2, LOW);
  digitalWrite(m3, HIGH);
  digitalWrite(m4, LOW);
  analogWrite(e1, mspeed);
  analogWrite(e2, mspeed);
  readline();
}

/*
 * Function Name: stopmotor()
 * Input: None
 * Output: None
 * Logic: This function stops the motors and activates a buzzer to signal motor shutdown.
 *        It achieves this by setting the motor control pins (m1, m2, m3, m4) to LOW, 
 *        setting the motor speed control pins (e1, e2) to 0 using analogWrite(),
 *        and setting the relay pin (RL) to HIGH. Additionally, it triggers a short beep using the beep() function.
 * Example Call: stopmotor();
 */
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
/*
 * Function Name: executeInstruction()
 * Input: char instruction - The movement instruction to be executed ('F' for forward, 'L' for left, 'R' for right)
 * Output: None
 * Logic: This function takes a movement instruction as input and performs the corresponding action.
 *        It uses a switch statement to execute the appropriate function based on the given instruction:
 *        - 'F': Calls the straight() function to move forward.
 *        - 'L': Calls the turnleft() function to turn left.
 *        - 'R': Calls the turnright() function to turn right.
 *        The function also prints the executed instruction to the Serial monitor.
 * Example Call: executeInstruction('F');
 */
void executeInstruction(char instruction) {
  switch (instruction) {
    case 'F':
      Serial.println("Moving forward");
      straight();
      break;
    case 'L':
      Serial.println("Turning left");
      turnleft();
      break;
    case 'R':
      Serial.println("Turning right");
      turnright();
      break;
    default:
      Serial.println("Invalid instruction");
      break;
  }
}

/*
 * Function Name: follow_line()
 * Input: None
 * Output: None
 * Logic: This function controls the bot to follow a black line using the infrared sensors.
 *        It reads the values from the five infrared sensors and adjusts the motor speeds based on the sensor readings.
 *        The adjustments are made to ensure that the bot stays on the black line:
 *        - If the middle sensor (s3) detects black, both motors are set to move forward.
 *        - If the left sensor (s2) detects black, the right motor (m1) is slowed down to turn right.
 *        - If the right sensor (s4) detects black, the left motor (m3) is slowed down to turn left.
 *        - If both the left and right sensors detect black, the bot is considered to be on a straight path.
 *        The function introduces a short delay to allow for sensor readings and motor adjustments.
 * Example Call: follow_line();
 */
void follow_line() {
  readline();
  // Follow the black line
  if (s3 == 1) {
    straight();
  } else if (s2 == 1) {
    correctleft();
  } else if (s4 == 1) {
    correctright();
  } else if (s2 == 1 && s4 == 1) {
    straight();
  }
  delay(5);
}

/*
 * Function Name: avoid_line()
 * Input: None
 * Output: None
 * Logic: This function implements a line avoidance mechanism using the infrared sensors.
 *        It reads the values from the five infrared sensors and adjusts the motor speeds based on the sensor readings.
 *        The adjustments are made to ensure that the bot avoids the black line:
 *        - If the middle sensor (s3) detects black, both motors are stopped, and the stopmotor() function is called.
 *        - If the left sensor (s2) detects black, the right motor (m1) is slowed down to turn right.
 *        - If the right sensor (s4) detects black, the left motor (m3) is slowed down to turn left.
 *        - If both the left and right sensors detect black, the bot is considered to be on a straight path.
 *        The function introduces a short delay to allow for sensor readings and motor adjustments.
 * Example Call: avoid_line();
 */
void avoid_line() {
  readline();
  // Avoid the black line
  if (s3 == 1) {
    stopmotor();
  } else if (s2 == 1) {
    correctleft();
  } else if (s4 == 1) {
    correctright();
  } else if (s2 == 1 && s4 == 1) {
    stopmotor();
  }
  delay(5);
}

/*
 * Function Name: beep()
 * Input: None
 * Output: None
 * Logic: This function activates a buzzer for a short duration to produce a beep sound.
 *        It achieves this by setting the buzzer pin (buzz) to HIGH for a short duration and then setting it back to LOW.
 *        The duration of the beep is controlled by the delay time.
 * Example Call: beep();
 */
void beep() {
  digitalWrite(buzz, HIGH);
  delay(50);
  digitalWrite(buzz, LOW);
}

/*
 * Function Name: lbeep()
 * Input: None
 * Output: None
 * Logic: This function activates a buzzer for a longer duration to produce a beep sound.
 *        It achieves this by setting the buzzer pin (buzz) to HIGH for a longer duration and then setting it back to LOW.
 *        The duration of the beep is controlled by the delay time.
 * Example Call: lbeep();
 */
void lbeep() {
  digitalWrite(buzz, HIGH);
  delay(500);
  digitalWrite(buzz, LOW);
}
