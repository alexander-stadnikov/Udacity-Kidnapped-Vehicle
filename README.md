# Particle Filter

This repository contains the implementation of the "Kidnapped Vehicle" Project for Udacity Course Self-Driving Car Engineer.

## Motivation

The precise localization is necessary for a self-driving vehicle.

The car might obtain its position with GPS. The main problem of GPS is accuracy. If the GPS signal is strong, it's possible to predict the car location with 1-3 meters accuracy. If the signal is weaker, then the accuracy drops to 10-50 meters, which entirely insufficient for a self-driving car and might lead to catastrophic consequences.

The [Particle Filter](https://en.wikipedia.org/wiki/Particle_filter) algorithm might solve the problem. The algorithm combines data from GPS and additional sensors, such as LIDAR, RADAR, and cameras. The accuracy might be raised to 3-10 cm.

## Brief overview of the algorithm

1. The car gets its rough position from GPS and works with a smaller region on the provided map. It's necessary to decrease the computational complexity. The map might either stored on the car or obtained from the Internet on demand
2. The map contains landmarks
3. The car recognizes some landmarks around itself and converts their position from the car coordinate system to the global
4. The Particle Filter estimates the car position

## Dependencies

* [The Udacity Self-Driving Car Engineer Nanodegree Simulator](https://github.com/udacity/self-driving-car-sim/releases)
* cmake >= 3.5
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
* gcc/g++ >= 5.4
* [uWebSocketIO](https://github.com/uWebSockets/uWebSockets)

## Build
The Particle Filter program can be built and run by doing the following from the project top directory.

```bash
$ ./build.sh
```

## Run
Run from the project top directory.

```bash
$ ./run.sh
```

## Result

After many experiments, I decided to use only 500 particles for the filter. The minimum possible amount is sufficient for the project, but the number might be easily increased for a real-life situation.

Finally, I got the approval from the simulator "Success! Your particle filter passed!". The result is below.

![Demo](./pf.gif)
