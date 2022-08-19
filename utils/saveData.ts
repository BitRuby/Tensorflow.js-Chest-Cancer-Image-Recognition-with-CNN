import fs from "fs";
import path from "path";
import * as tf from "@tensorflow/tfjs-node";

export const saveTensor = (tensor: tf.Tensor, path: string) => {
  try {
    fs.writeFileSync(
      ensureDirectoryExistence(path),
      JSON.stringify(tensor.arraySync())
    );
  } catch (err) {
    console.error(err);
  }
};

const ensureDirectoryExistence = (filePath: string) => {
  var dirname = path.dirname(filePath);
  if (fs.existsSync(dirname)) {
    return filePath;
  }
  ensureDirectoryExistence(dirname);
  fs.mkdirSync(dirname);
  return filePath;
};
