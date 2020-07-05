g++ plugin_yolov3_detection_op_test.cc \
  -I ${NEUWARE}/include  \
  -I ../common/include  \
  -L ${NEUWARE}/lib64 \
  -L ../build \
  -o ./yolov3_detection_test -lcnml -lcnrt -lcnplugin --std=c++11
