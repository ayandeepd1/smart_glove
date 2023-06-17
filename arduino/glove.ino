#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "models.h"

#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps612.h"

#include "FS.h"
#include "SD.h"
#include "SPI.h"

static SemaphoreHandle_t mutex;

MPU6050 mpu;

const uint8_t SET_PIN=13;
const uint8_t INTERRUPT_PIN=4; 
const uint8_t LED_PIN=2;
const int16_t n_samples=1;

int16_t packet[n_samples][14];

bool blinkState = false;

bool dmpReady = false;  // set true if DMP init was successful
uint8_t mpuIntStatus;   // holds actual interrupt status byte from MPU
uint8_t devStatus;      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[64]; // FIFO storage buffer

int16_t quaternion[4];  // [w, x, y, z]         quaternion container
uint16_t v[10];         //should be locked in mutex


//uint8_t p[]={14, 27, 26, 25, 33, 32, 35, 34, 39, 36};
uint8_t p[]={36, 39, 34, 35, 32, 33, 25, 26, 27, 14};

long t1, t2;

void TaskPrint(void *pvParameters);
void TaskDetermine(void *pvParameters);

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 25*1024;
uint8_t tensor_arena[kTensorArenaSize];
}

volatile bool mpuInterrupt = false;   
void dmpDataReady() {
  mpuInterrupt = true;
}

void setup_mpu(){
  Wire.begin();
  Wire.setClock(4000000);
  
  mpu.initialize();
  Serial.println(mpu.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));
    
    mpu.setXGyroOffset(-59);
    mpu.setYGyroOffset(-114);
    mpu.setZGyroOffset(63);
    mpu.setXAccelOffset(-1898);
    mpu.setYAccelOffset(984);
    mpu.setZAccelOffset(1794);

  devStatus = mpu.dmpInitialize();
  if (devStatus == 0) {
    mpu.CalibrateAccel(6);
    mpu.CalibrateGyro(6);
      
    Serial.println();
    mpu.PrintActiveOffsets();
    Serial.println(F("Enabling DMP..."));
    mpu.setDMPEnabled(true);
    Serial.print(F("Enabling interrupt detection (Arduino external interrupt "));
    Serial.print(digitalPinToInterrupt(INTERRUPT_PIN));
    Serial.println(F(")..."));
    attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
    mpuIntStatus = mpu.getIntStatus();
    Serial.println(F("DMP ready! Waiting for first interrupt..."));
    dmpReady = true;
    packetSize = mpu.dmpGetFIFOPacketSize();
  } else {
    Serial.print(F("DMP Initialization failed (code "));
    Serial.print(devStatus);
    Serial.println(F(")"));
  }
}

void setup_sdcard(){
  if(!SD.begin()){
        Serial.println("Card Mount Failed");
        return;
    }
    uint8_t cardType = SD.cardType();
    if(cardType == CARD_NONE){
        Serial.println("No SD card attached");
        return;
    }
    uint64_t cardSize = SD.cardSize() / (1024 * 1024);
    Serial.printf("SD Card Size: %lluMB\n", cardSize);
}

void setup_gpio(){ 
   
  pinMode(SET_PIN, INPUT_PULLUP);
  //if(digitalRead(SET_PIN)==0)
    //#define OUT_SERIAL
  pinMode(LED_PIN, OUTPUT);
  pinMode(INTERRUPT_PIN, INPUT);
  for(int i=0; i<10; i++)
    pinMode(p[i], INPUT);
}

void listDir(fs::FS &fs, const char * dirname, uint8_t levels){
    Serial.printf("Listing directory: %s\n", dirname);
    File root = fs.open(dirname);
    if(!root){
        Serial.println("Failed to open directory");
        return;
    }
    File file = root.openNextFile();
    while(file){
        Serial.print(file.name());
        Serial.print("  ");
        Serial.println(file.size());
        file = root.openNextFile();
    }
}
void setup() {
  Serial.begin(115200);
  while (!Serial); 
  //setup_tflite();
  setup_gpio();
  setup_mpu();
  setup_sdcard();
  
  mutex = xSemaphoreCreateMutex();

  xTaskCreatePinnedToCore(
    TaskDetermine,    //function name
    "TaskDetermine",  // A name just for humans
    50*1024,  // This stack size can be checked & adjusted by reading the Stack Highwater
    NULL,  // parameters
    0,     // Priority, with 3 (configMAX_PRIORITIES - 1) being the highest, and 0 being the lowest.
    NULL,  // task handle callback
    0);   //  core id
  xTaskCreatePinnedToCore(
    TaskRecord,    //function name
    "TaskRecord",  // A name just for humans
    2096,  // This stack size can be checked & adjusted by reading the Stack Highwater
    NULL,  // parameters
    3,     // Priority, with 3 (configMAX_PRIORITIES - 1) being the highest, and 0 being the lowest.
    NULL,  // task handle callback
    1);   //  core id
    
} 

void loop() {  
delay(1000);
}
void TaskRecord(void *pvParameters){
  int c=0;
  while(1){
  t1=micros();
  digitalWrite(LED_PIN, 1);
    //for(int c=0; c<n_samples; c++){
      mpu.dmpGetCurrentFIFOPacket(fifoBuffer);
      for(int i=0; i<10; i++)
        v[i]=analogRead(p[i]);
      xSemaphoreTake(mutex, portMAX_DELAY);
      mpu.dmpGetQuaternion(&packet[c][10], fifoBuffer); 
      memcpy(&packet[c][0], &v[0], 10*2);
      xSemaphoreGive(mutex);
    //}
  digitalWrite(LED_PIN, 0);
  t2=micros();
  //Serial.printf("time=%d\n", t2-t1);
  vTaskDelay(100);
  }  
}

void TaskDetermine(void *pvParameters){
  
  String outcomes[]={"hi", "thumbsup", "point_index", "ok", "peace", "call"};
  
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_data);
  static tflite::AllOpsResolver resolver;
  /**
  static tflite::MicroInterpreter static_interpreter( model, 
                                                      resolver, 
                                                      tensor_arena,
                                                      kTensorArenaSize, 
                                                      error_reporter);
  interpreter = &static_interpreter;
  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);
  **/
  for(;;){
    for(int i=0; i<n_samples; i++){
      for(int j=0; j<14; j++){
        xSemaphoreTake(mutex, portMAX_DELAY);
        //input->data.f[j] = test[i][j];
        int val=packet[i][j];
        xSemaphoreGive(mutex);
        //Serial.printf("%d,", val);
        //#ifdef OUT_SERIAL
          //Serial.printf("%d,", packet[i][j]);
      }
    //Serial.println();
  }
  vTaskDelay(100);
  }
}
