# OpenMV - Gesture Classification from Static Image
# Author: Abhishek Pandey

import sensor, time, ml, uos, gc, image

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((240, 240))
sensor.skip_frames(time=2000)

# Load the model
net = None
labels = None

try:
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite": ' + str(e))

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt": ' + str(e))

# Load the test image (must be 240x240 RGB BMP)
try:
    img = image.Image("gesture1.bmp", copy_to_fb=True)
except Exception as e:
    raise Exception("Failed to load image: " + str(e))

# Run inference
clock = time.clock()
clock.tick()
start_time = time.ticks_ms()

# Run prediction
predictions = net.predict([img])[0].flatten().tolist()

inference_time = time.ticks_diff(time.ticks_ms(), start_time)

# Output results
print("Gesture classification result:")
for i, score in enumerate(predictions):
    print("%s = %.2f%%" % (labels[i], score * 100.0))

print("Inference time: %d ms" % inference_time)
