import fs from "fs";
import path from "path";
import { loadTensor } from "./utils/loadData";
import CNN from "./cnn";
import { convertAll } from "./utils/imagesToTensors";
import * as tf from "@tensorflow/tfjs-node";
import { labelEncoder, normalise, trainTestSplit } from "./utils/preprocessing";

const loadDataset = (source: string) => {
  console.log(`Reading tensors from ${source}...`);
  const features: any[] = [],
    labels: string[] = [];
  fs.readdirSync(source).forEach((subdir) => {
    fs.readdirSync(path.join(source, subdir)).forEach((file) => {
      features.push(loadTensor(path.join(source, subdir, file)));
      labels.push(subdir);
    });
  });
  return [features, labels];
};


const [trainFeatures, trainLabels] = loadDataset("./TensorData/train");
const [testFeatures, testLabels] = loadDataset("./TensorData/test");
const [validFeatures, validLabels] = loadDataset("./TensorData/valid");


const cnn = new CNN({
  imageWith: 32,
  imageHeight: 32,
  channels: 3,
  numOutputs: 4,
  batchSize: 32,
  epochs: 50
});

const trainFeaturesTensor = tf.tensor(trainFeatures, [trainFeatures.length, 32, 32, 3]);
const normalisedTrainFeaturesTensor = normalise(trainFeaturesTensor).tensor;
labelEncoder(trainLabels);
const trainLabelsTensor = tf.oneHot(trainLabels, 4)

const testFeaturesTensor = tf.tensor(testFeatures, [testFeatures.length, 32, 32, 3]);
const normalisedTestFeaturesTensor = normalise(testFeaturesTensor).tensor;
labelEncoder(testLabels);
const testLabelsTensor = tf.oneHot(testLabels, 4)

cnn.train(normalisedTrainFeaturesTensor, trainLabelsTensor, normalisedTestFeaturesTensor, testLabelsTensor);

