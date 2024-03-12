# Traffic monitoring system implementation

An implementation of a traffic monitoring system for counting incoming and outgoing vehicles.

<p align="center">
<video src="https://github.com/vladserkoff/detect-and-track/assets/9671366/3c9e86f7-3ea8-473a-9199-1c206b2bde2b" controls width="80%"></video>
</p>

Slightly more formal and boring description of the system can be found in the [system description document](./docs/system_description.md).

Hopefully more interesting is the [presentation](./docs/presentation.html) which, unfortunately, is not rendered by GitHub and has to be downloaded and opened locally.

## Usage

### Docker

The easiest way is to use the provided `docker-compose.yml` file. By default it expects that the machine has an Nvidia GPU and docker is configured to use it. If that is not the case, comment out `deploy` section of the `docker-compose.yml` and append `"--device", "cpu"` to the `command` field.

```bash
docker-compose up
```

This will process the video from `data/Video.mp4` and save the result to `outputs/Video.mp4`.

### Local

Local installation allows for more interactive usage when the video is processed in real time on the screen. This requires first installing the dependencies:

```bash
conda env create -f environment.yml
conda activate somecompany
```

Then the program can be run with:

```bash
python -m main -s ./path/to/Video.mp4
```

### Tests

Tests can be (**successfully!**) run with:

```bash
python -m unittest discover tests
```

## Technical details

This program is written in **python** and uses **pytorch** for deep learning and **opencv** for computer vision. Models are provided by **huggingface** and **ultralytics** packages. For a more exhaustive list of dependencies see [environment.yml](./environment.yml).
