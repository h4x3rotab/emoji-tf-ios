# Emoji TensorFlow-iOS

This is a TensorFlow demo that can be run on iOS. It implements a text classifier
that can predict emoji from short text (like tweets).

Presentation: `TensorFlow on iOS.pdf`

## License

[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](http://creativecommons.org/licenses/by-nc/4.0/)


## Prerequests

* [TensorFlow (installation & source code)](https://www.tensorflow.org)
* [Keras](https://keras.io)
* Python 3
* Xcode
* Jupyter (for viewing ipynb files)
* [Twitter data](https://archive.org/search.php?query=twitterstream)

## How to train the model?

I have included a pretrained Keras model in this repository (p5-40-test.hdf5) that
you can play with. But in case you want to train it by yourself, here is a brief
guide.

1. Prepare training data.
    1. Downlaod and unzip twitter training data.
    2. Modify the `$INPUT` directory in `extract_all.sh` and run the script. Then you will get `data/extracted.list`.
    3. Run `stats_top.py` to get the top emojis stored at `data/stat.txt`.
    4. Open Jupyter notebook and run `build_dataset.ipynb`. It produces `data/dataset.pickle` with all sampled training data.
    5. Run `tokenize_dataset.ipynb` to produce the tokenized dataset `data/plain_dataset.pickle` as well as metadata `data/plain_dataset_meta.pickle`.
2. Run `train.py` to train the model. High end Cuda GPUs are recommended. The trained Keras model will be saved as `p5-40-test.hdf5`.
3. (Optional) You can try the model on arbitrary input with `replayer.ipynb`.
4. (Optional) You can also try to visualize the training process by `tensorboard --log_dir=.` if you have trained a model.

## How to compile TensorFlow for iOS?

Follow the official compile guide [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/ios_examples).

## How to run the model on iOS?

Unfortunately, TensorFlow for iOS is still an alpha version. So we have to tweak a
little bit to make it work.

### Add additional integer Ops for LSTM.

Navigate to `tensorflow/core/kernels` directory and change the code like this:

At `cwise_op_add_1.cc`:

        // -- Original
        REGISTER5(BinaryOp, CPU, "Add", functor::add, float, Eigen::half, double, int32,
                  int64);
        #if TENSORFLOW_USE_SYCL

        // -- Change to
        REGISTER5(BinaryOp, CPU, "Add", functor::add, float, Eigen::half, double, int32,
                  int64);
        #if defined(__ANDROID_TYPES_SLIM__)
        REGISTER(BinaryOp, CPU, "Add", functor::add, int32);
        #endif  // __ANDROID_TYPES_SLIM__
        #if TENSORFLOW_USE_SYCL

At `cwise_op_less.cc`:

        // -- Original
        REGISTER8(BinaryOp, CPU, "Less", functor::less, float, Eigen::half, double,
                  int32, int64, uint8, int8, int16);
        #if GOOGLE_CUDA

        // -- Change to
        REGISTER8(BinaryOp, CPU, "Less", functor::less, float, Eigen::half, double,
                  int32, int64, uint8, int8, int16);
        #if defined(__ANDROID_TYPES_SLIM__)
        REGISTER(BinaryOp, CPU, "Less", functor::less, int32);
        #endif  // __ANDROID_TYPES_SLIM__
        #if GOOGLE_CUDA

Then compile TensorFlow again and you won't encounter "No OpsKernel found" issue.

### Convert the Keras model to TensorFlow model.

Run `export_tf_model.ipynb` to convert Keras model file `p5-40-test.hdf5` to TensorFlow
model:

* GraphDef: `export/p5-40-test-serving/graph-serving.pb`
* Checkpoint: `export/p5-40-test-serving/model-ckpt-*`

Navigate to `export/p5-40-test-serving` directory and run the following command to
convert the model to mobile version:
        python3 -m tensorflow.python.tools.freeze_graph \
          --input_graph="graph-serving.pb" --input_checkpoint="model-ckpt" \
          --output_graph="frozen.pb" --output_node_names="dense_2/Softmax"

Finally you will get `forzen.pb` file which will be used later.

### Run the model on iOS.

Copy the Xcode project `emoji_demo` to `tensorflow/tensorflow/contrib/ios_examples`.
You should be able to compile and run it on iOS now. The demo itself includes a
pretrained model at `data/forzen.pb`. To run your own model, you have to replace it
with yours.

