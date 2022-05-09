# Rapper-look-like: find out what rapper you look like!

## How to run
1. Install and run [Docker](https://www.docker.com/)
2. Build Docker image using `docker build -t <your-image-name> .`
3. Run Docker container using `docker run -v ${pwd}/data/images/:/root/rapper-face-similarity/data/images -v ${pwd}/config/config.yaml:/root/rapper-face-similarity/config/config.yaml <your-image-name>` 

## Example
1. Upload images you'd to preidct to `data/images/inference/` folder
2. Run docker container
3. Results are saved in `results.png` file
