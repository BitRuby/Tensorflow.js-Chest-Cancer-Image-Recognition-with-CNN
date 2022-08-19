import * as tf from "@tensorflow/tfjs-node";

interface NetworkOptions {
  imageWith: number;
  imageHeight: number;
  channels: number;
  numOutputs: number;
  batchSize: number;
  epochs: number;
}

export default class CNN {
  options: NetworkOptions;
  model: tf.Sequential = tf.sequential();
  trainFeatures: any;
  trainLabels: any;
  testFeatures: any;
  testLabels: any;

  constructor(options?: NetworkOptions) {
    this.options = Object.assign(
      {
        imageWith: 32,
        imageHeight: 32,
        channels: 3,
        numOutputs: 4,
        batchSize: 32,
        epochs: 100,
      },
      options
    );
    this.setModel();
  }

  getOptions() {
    return this.options;
  }

  setModel() {
    this.model.add(
      tf.layers.conv2d({
        inputShape: [
          this.options.imageWith,
          this.options.imageHeight,
          this.options.channels,
        ],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      })
    );
    this.model.add(
      tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
    );
    this.model.add(tf.layers.flatten());
    this.model.add(
      tf.layers.dense({
        units: this.options.numOutputs,
        kernelInitializer: "varianceScaling",
        activation: "softmax",
      })
    );
    const optimizer = tf.train.adam();
    this.model.compile({
      optimizer: optimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
  }

  getModel() {
    return this.model;
  }

  async train(trainFeatures: any, trainLabels: any) {
    this.trainFeatures = trainFeatures;
    this.trainLabels = trainLabels;

    return this.model.fit(this.trainFeatures, this.trainLabels, {
      epochs: this.options.epochs,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, log) =>
          console.log(`Epoch ${epoch}, loss: ${log?.loss}`),
      },
    });
  }

  test() {}

  predict() {}
}
