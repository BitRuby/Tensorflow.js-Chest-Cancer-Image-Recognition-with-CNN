import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";

export const loadImage = (path: string, resize: [number, number]) => {
  const image = fs.readFileSync(path);

  return tf.tidy(() => {
    const tensor = tf.node.decodeImage(image, 3);
    return resize ? tf.image.resizeBilinear(tensor, resize) : tensor;
  });
};

export const loadTensor = (path: string) => {
  try {
    const tensor = fs.readFileSync(path);
    const tensorAsArray = JSON.parse(tensor.toString());
    return tensorAsArray;
  } catch (err) {
    console.error(err);
  }
};
