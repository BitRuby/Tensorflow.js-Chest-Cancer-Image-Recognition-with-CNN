import * as tf from "@tensorflow/tfjs-node";

export const normalise = (tensor: tf.Tensor) => {
  const max = tensor.max();
  const min = tensor.min();
  const normalisedTensor = tensor.sub(min).div(max.sub(min));
  return { tensor: normalisedTensor, min, max };
};

export const labelEncoder = (labels: any[]) => {
  const labelSet = [...new Set(labels)];
  labels.forEach((e, i) => labels[i] = labelSet.indexOf(e) + 1);
};

export const trainTestSplit = (features: any[], labels: any[], numSplit: [number, number]) => {
  const featuresBounds = Math.floor(features.length * numSplit[0] / (numSplit[0] + numSplit[1]));
  const labelsBounds = Math.floor(labels.length * numSplit[0] / (numSplit[0] + numSplit[1]));
  return [features.splice(0, featuresBounds), labels.splice(0, labelsBounds), features, labels];
}