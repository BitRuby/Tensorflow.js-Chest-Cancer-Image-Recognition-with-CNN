import fs from "fs";
import path from "path";
import { loadTensor } from "./utils/loadData";
import CNN from "./cnn";
import { convertAll } from "./utils/imagesToTensors";
import * as tf from "@tensorflow/tfjs-node";
import { labelEncoder, normalise } from "./utils/preprocessing";

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

convertAll([16,16]);

const [trainFeatures, trainLabels] = loadDataset("./TensorData/train");
const [testFeatures, testLabels] = loadDataset("./TensorData/test");
const [validFeatures, validLabels] = loadDataset("./TensorData/valid");

const cnn = new CNN({
  imageWith: 16,
  imageHeight: 16,
  channels: 3,
  numOutputs: 4,
  batchSize: 32,
  epochs: 50
});

labelEncoder(trainLabels);
labelEncoder(testLabels);
const trainFeaturesTensor = tf.tensor(trainFeatures, [trainFeatures.length, 16, 16, 3]);
const normalisedTrainFeaturesTensor = normalise(trainFeaturesTensor).tensor;
const trainLabelsTensor = tf.oneHot(trainLabels, 4)
cnn.train(normalisedTrainFeaturesTensor, trainLabelsTensor);



