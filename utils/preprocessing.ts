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
