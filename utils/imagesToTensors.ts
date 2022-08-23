import fs from "fs";
import path from "path";
import { loadImage } from "./loadData";
import { saveTensor } from "./saveData";

export const imagesToTensors = (
  source: string,
  destination: string,
  shape: [number, number] = [32, 32]
) => {
  console.log(
    `Converting images from ${source} to tensors and saving on ${destination}...`
  );
  fs.readdirSync(source).forEach((subdir) => {
    fs.readdirSync(path.join(source, subdir)).forEach((file) => {
      const tensor = loadImage(path.join(source, subdir, file), shape);
      saveTensor(
        tensor,
        path.join(destination, subdir, file.replace(".png", ""))
      );
    });
  });
};

export const convertAll = (shape: [number, number]) => {
  imagesToTensors("./Data/train", "./TensorData/train", shape);
  imagesToTensors("./Data/test", "./TensorData/test", shape);
  imagesToTensors("./Data/valid", "./TensorData/valid", shape);
};