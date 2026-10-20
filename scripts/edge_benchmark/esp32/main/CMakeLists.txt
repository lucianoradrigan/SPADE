# Requires the esp-tflite-micro component (idf.py add-dependency "espressif/esp-tflite-micro").
idf_component_register(SRCS "main.cc" "model_data.cc" INCLUDE_DIRS ".")
