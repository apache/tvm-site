TODO Replace all \bfloat{}

# Bring Your Own Datatypes: Enabling Custom Datatype Exploration in TVM

In this post, we describe the Bring Your Own Datatypes framework, which enables the use of custom datatypes within TVM.

## Introduction

When designing accelerators, an important decision is how one will approximately represent real numbers in hardware.
This problem has had a longstanding, industry-standard solution: the IEEE 754 floating-point standard [TODO cite num][ieee754].
When trying to squeeze the most out of hardware, though, will we necessarily use IEEE 754 floats?
There are a number of reasons to believe we could build a better datatype: one which is smaller, faster, or more power efficient, and overall more optimal for our workload and the constraints of our hardware design.

Researchers have already begun experimenting with new datatypes in academic and industrial accelerator designs.
For example, Google's Tensor Processing Unit (the TPU) uses the \bfloat{} type: a single-precision IEEE float which has been truncated to 16 bits, instantly reducing storage cost by half while often losing no model accuracy [TODO cite num][jouppi2017datacenter].
Before researchers begin building hardware for their datatype, however, they first need to determine how their datatype will behave numerically in the workloads they care about.
This often involves first building a software-emulated version of their datatype, and then hacking the datatype directly into workloads;
even better is to integrate the datatype directly into compilers themselves.
Both routes can be tedious, with the latter route often becoming unmanageable given the size and complexity of modern compilers.
One example taken from GitHub shows someone hacking the *posit* datatype into TensorFlow [TODO cite num][posittensorflow].
The result is 237 commits, adding nearly 6000 lines of code and touching over 200 files across the codebase---and that's just to add one datatype!
This amount of work is prohibitive for many researchers.

To address these problems, we present the Bring Your Own Datatypes framework.
The framework enables easy exploration of new datatypes in deep learning workloads by allowing users to plug their simulated datatype into TVM.
Unlike the posits-in-Tensorflow example above, which enables a single new datatype in a compiler, the Bring Your Own Datatype framework enables a huge variety of user-defined types.
We also show how the framework can be used to conduct valuable datatype research and exploration.

In the rest of this blog post, we first describe the design and implementation of the framework, and how it integrates with TVM.
We then show an example usage of the framework by conducting a preliminary study of how changing datatypes affect pretrained models.

## References
[ieee754]: TODO
[jouppi2017datacenter]: TODO
[posittensorflow]: TODO
