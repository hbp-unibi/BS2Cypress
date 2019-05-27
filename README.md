# BS2Cypress
Cypress Backend for BrainScaleS written in C++

_______
## Overview
This library provides a backend implementation of the BrainScaleS (BS) system [1] for the Cypress library [2]. The current workflow makes use of runtime dynamic loading of the pre-compiled library. The benefits of this approach are: 

 * The user is not required to provided the full BrainScaleS software stack at built time
 * The backend is not included in a statically linked library
 * Since the BS2Cypress library is dynamically linked, BS dependencies are resolved at runtime

A disadvantage is, that the library has to be in the same folder as the executable at runtime. The cypress library will take care of it if you are using its NMPI interface.

_______
## Structure

In `backend/` you will find the actual backend implementation, while `test/` provides some unit tests. 'bin/' contains the pre-compiled library.
The `setup_includes.sh` is automatically executed at build time, setting up include dependencies, as these are currently not deployed on the BrainScaleS server. 




[1] See e.g. Schmitt, Sebastian, et al. "Neuromorphic hardware in the loop: Training a deep spiking network on the brainscales wafer-scale system." 2017 International Joint Conference on Neural Networks (IJCNN). IEEE, 2017. https://arxiv.org/abs/1703.01909  
[2] https://github.com/hbp-unibi/cypress
