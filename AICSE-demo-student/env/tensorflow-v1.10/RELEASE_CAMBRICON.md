# Release 0.9.0
Initial release of Cambricon TensorFlow based on original
TensorFlow-v1.10.0 (commit: 656e7a2b347c3c6eb76a6c130ed4b1def567b6c1).
## Major Features
* Add MLU device support based on TensorFlow-v1.10.0.
  MLU devices
  (1) support fp16 and qint8 (representing float tensor with
      int8 numbers, an int position and a float scale)
      computation.
  (2) accelerate sparse convolution and full connection
      via hardware support.
* Add several MLU kernels for inference.
* Support two execution modes for MLU sub-graphs:
  (1) kernels are executed one by one, and
  (2) kernels are fused to enable cross-kernel compile
      optmization and reduce kernel launches.
* MLU supports two ways of model deployment:
  (1) using TF origin *.pb to execute graph, and
  (2) converting *.pb to *.cambricon, and
      executing graph using cambricon model.
* Support both (1) model parallelism
  (i.e., a batch of input is processed by several MLU cores)
  and (2) data parallelism
  (i.e., each MLU core process a batch of inputs).
* Support model parallelism on multiple MLU devices
  (use ResNet18 as a demo).
* Add several MLU demos:
  (1) image classification: AlexNet, VGG16, VGG19,
      ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
      MobileNet, SqueezeNet, InceptionV1 and InceptionV3.
  (2) object detection: Faster-RCNN (w/ InceptionResNetV2)
      and SSD (w/ MobileNetV1).
  (3) image classification models in Ai-Matrix benchmark.

-------------------------------------------------------------------------------------------------------
model            | dense-fp16 | dense-int8 | sparse-fp16 | sparse-int8 | model-parallel | data-parallel
-------------------------------------------------------------------------------------------------------
AlexNet            Y            Y            Y             Y             Y                Y
VGG16              Y            Y            Y             Y             Y                Y
VGG19              Y            Y            Y             Y             Y                Y
ResNet18           Y            Y            Y             Y             Y                Y
ResNet34           Y            Y            Y             Y             Y                Y
ResNet50           Y            Y            Y             Y             Y                Y
ResNet101          Y            Y            Y             Y             Y                Y
ResNet152          Y            Y            Y             Y             Y                Y
MobileNet          Y            Y            Y             N             Y                Y
SqueezeNet         Y            Y            Y             N             Y                Y
InceptionV1        Y            Y            Y             Y             Y                Y
InceptionV3        Y            Y            Y             Y             Y                Y
FasterRCNN-                                                                                
InceptionResNetV2  Y            N            N             N             N                Y
SSD-MobileNetV1    Y            N            N             N             N                Y
