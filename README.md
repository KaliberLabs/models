# Kaliber Labs fork of Tensor Flow Models

It contains tools for detecting people and objects. 

* models/ contains tensorflow models and tools
* models/app/ contains wrappers around tensorflow models for use in applications

Tools

* models/app/detect_humans.py is a CLI tool that detects all the humans in a frame and writes to CSV.
* models/app/draw_rects.py is an UNFINISHED CLI tool to draw rectangles around people in jpegs
* models/app/model_download.py is a small program that downloads a TF model
* models/app/api.py is an UNTESTED, NEVER USED REST wrapper around the detection model
* models/app/humans.py is a library module that provides a usable abstraction around tensorflow code


### Running the people detection model server

Copy a tensorflow model

    cd ~
    clone this repo into $HOME
    cd models/object_detection
    aws s3 cp s3://kaliber-face-experiments/es/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017_people_detector.pb .
    cd ../app

Check that in config.py are correct.


    python app/api.py  

    If you get an Out-Of-Memory (OOM) error, increase the GPU size or run the model on RAM like this:

    CUDA_VISIBLE_DEVICES="" python app/api.py  

  
You can submit jpg images  like this:

    ⋊> ~/Pictures curl -v --data-binary '@ejaj.jpg' 'http://ec2-52-90-84-45.compute-1.amazonaws.com:8888/' -v                                                                                                                                                         14:20:58
    *   Trying 52.90.84.45...
    * Connected to ec2-52-90-84-45.compute-1.amazonaws.com (52.90.84.45) port 8888 (#0)
    > POST / HTTP/1.1
    > Host: ec2-52-90-84-45.compute-1.amazonaws.com:8888
    > User-Agent: curl/7.47.0
    > Accept: */*
    > Content-Length: 14931
    > Content-Type: application/x-www-form-urlencoded
    > Expect: 100-continue
    > 
    < HTTP/1.1 100 Continue
    * We are completely uploaded and fine
    ^[* HTTP 1.0, assume close after body
    < HTTP/1.0 200 OK
    < Content-Type: application/json
    < Content-Length: 173
    < Server: Werkzeug/0.12.1 Python/3.6.1
    < Date: Tue, 19 Sep 2017 21:22:03 GMT
    < 
    [
      {
        "box": [
          0.012828623875975609, 
          0.0574076846241951, 
          0.9920268058776855, 
          0.9830871820449829
        ], 
        "score": 0.9998437166213989
      }
    ]
    * Closing connection 0


------------------------------

# TensorFlow Models

This repository contains machine learning models implemented in
[TensorFlow](https://tensorflow.org). The models are maintained by their
respective authors. To propose a model for inclusion, please submit a pull
request.

Currently, the models are compatible with TensorFlow 1.0 or later. If you are
running TensorFlow 0.12 or earlier, please
[upgrade your installation](https://www.tensorflow.org/install).


## Models
- [adversarial_crypto](adversarial_crypto): protecting communications with adversarial neural cryptography.
- [adversarial_text](adversarial_text): semi-supervised sequence learning with adversarial training.
- [attention_ocr](attention_ocr): a model for real-world image text extraction.
- [audioset](audioset): Models and supporting code for use with [AudioSet](http://g.co.audioset).
- [autoencoder](autoencoder): various autoencoders.
- [cognitive_mapping_and_planning](cognitive_mapping_and_planning): implementation of a spatial memory based mapping and planning architecture for visual navigation.
- [compression](compression): compressing and decompressing images using a pre-trained Residual GRU network.
- [differential_privacy](differential_privacy): privacy-preserving student models from multiple teachers.
- [domain_adaptation](domain_adaptation): domain separation networks.
- [im2txt](im2txt): image-to-text neural network for image captioning.
- [inception](inception): deep convolutional networks for computer vision.
- [learning_to_remember_rare_events](learning_to_remember_rare_events):  a large-scale life-long memory module for use in deep learning.
- [lfads](lfads): sequential variational autoencoder for analyzing neuroscience data.
- [lm_1b](lm_1b): language modeling on the one billion word benchmark.
- [namignizer](namignizer): recognize and generate names.
- [neural_gpu](neural_gpu): highly parallel neural computer.
- [neural_programmer](neural_programmer): neural network augmented with logic and mathematic operations.
- [next_frame_prediction](next_frame_prediction): probabilistic future frame synthesis via cross convolutional networks.
- [object_detection](object_detection): localizing and identifying multiple objects in a single image.
- [real_nvp](real_nvp): density estimation using real-valued non-volume preserving (real NVP) transformations.
- [rebar](rebar): low-variance, unbiased gradient estimates for discrete latent variable models.
- [resnet](resnet): deep and wide residual networks.
- [skip_thoughts](skip_thoughts): recurrent neural network sentence-to-vector encoder.
- [slim](slim): image classification models in TF-Slim.
- [street](street): identify the name of a street (in France) from an image using a Deep RNN.
- [swivel](swivel): the Swivel algorithm for generating word embeddings.
- [syntaxnet](syntaxnet): neural models of natural language syntax.
- [textsum](textsum): sequence-to-sequence with attention model for text summarization.
- [transformer](transformer): spatial transformer network, which allows the spatial manipulation of data within the network.
- [tutorials](tutorials): models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).
- [video_prediction](video_prediction): predicting future video frames with neural advection.
