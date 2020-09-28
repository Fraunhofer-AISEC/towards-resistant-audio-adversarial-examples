
[Accompanying Website](https://tom-doerr.github.io/website_towards_resistant_audio_adversarial_examples/)

To generate adversarial examples for your own files, follow the below process
and modify the arguments to attack,py. Ensure that the file is sampled at
16KHz and uses signed 16-bit ints as the data type. You may want to modify
the number of iterations that the attack algorithm is allowed to run.


## Setup
1. Install Docker.
On Ubuntu/Debian/Linux-Mint etc.:
```
sudo apt-get install docker.io
sudo systemctl enable --now docker
```
Instructions for other platforms:
https://docs.docker.com/install/


2. Download DeepSpeech and build the Docker images:
```
./setup.sh
```

### With Nvidia-GPU support:
3. Install the NVIDIA Container Toolkit.
This step will only work on Linux and is only necessary if you want GPU support.
As far as I know it's not possible to use a GPU with docker under Windows/Mac.
On Ubuntu/Debian/Linux-Mint etc. you can install the toolkit with the following commands:
```sh
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
Instructions for other platforms (CentOS/RHEL):
https://github.com/NVIDIA/nvidia-docker

4. Start the container using the GPU image we just build:
```
docker run --gpus all -it --mount src=$(pwd),target=/audio_adversarial_examples,type=bind -w /audio_adversarial_examples aae_deepspeech_041_gpu
```

### CPU-only (Skip if already started with Nvidia-GPU support):
4. Start the container using the CPU image we just build:
```
docker run -it --mount src=$(pwd),target=/audio_adversarial_examples,type=bind -w /audio_adversarial_examples aae_deepspeech_041_cpu
```


## Basic usage
Classify audio `sample-000000.wav`
```
python3 classify.py sample-000000.wav
```

Generate an adversarial example
```
python3 attack.py --in sample-000000.wav --target "this is a test" --out adv.wav --iterations 1000 --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
```

Add offset to audio file `adv.wav`
```
python3 add_silence_to_start.py adv.wav
```

Classify audio file `adv.wav` with an added offset of 121 samples
```
python3 classify.py offset_audio_added/adv/121.wav
```

Classify all audio offset files for `adv.wav`
```
python3 classify.py offset_audio_added/adv/*.wav
```

Plot the edit distance for different offsets. Run the following command outside of the docker container (you might need to run `pip install -r requirements.txt` first):
```
python3 plot.py offset_audio_added/adv/
```

The code in this repo is based on the code from [Audio Adversarial Examples](https://github.com/carlini/audio_adversarial_examples). 
