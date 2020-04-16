---
layout: post
title:  "Bring Your Own Datatypes: Enabling Custom Datatype Exploration in TVM"
date:   2020-04-15
author: Gus Smith
---
- TODO Replace all \bfloat{}
- TODO Update date in heading
- TODO Update filename date

In this post, we describe the Bring Your Own Datatypes framework, which enables the use of custom datatypes within TVM.

## Introduction

When designing accelerators, an important decision is how one will approximately represent real numbers in hardware.
This problem has had a longstanding, industry-standard solution: the IEEE 754 floating-point standard [TODO cite num][ieee754].
When trying to squeeze the most out of hardware, though, will we necessarily use IEEE 754 floats?
There are a number of reasons to believe we could build a better datatype: one which is smaller, faster, or more power efficient, and overall more optimal for our workload and the constraints of our hardware design.

Researchers have already begun experimenting with new datatypes in academic and industrial accelerator designs.
For example, Google's Tensor Processing Unit (the TPU) uses the \bfloat{} type: a single-precision IEEE float which has been truncated to 16 bits, instantly reducing storage cost by half while often losing no model accuracy [TODO cite num][jouppi2017datacenter].
Before researchers begin building hardware for their datatype, however, they first need to determine how their datatype will behave numerically in the workloads they care about.
[TODO could add a sentence or two here convincing people that software-emulated versions of datatypes DO exist]
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

## Bring Your Own Datatypes

The goal of the Bring Your Own Datatypes framework
  is to enable users to run deep learning workloads
  using custom datatypes.

**Scope of datatypes supported:**
**What it doesn't support:**

The primary benefit of TVM over other frameworks is that it already abstracts away datatypes in a ``clean'' way.
That is, a few fundamental numeric types (\texttt{int}s and \texttt{float}s of various sizes) are defined in the framework, and are abstracted away in program representations.
Every node in a TVM program has an attached \texttt{dtype}, which can be any of the above types.
The framework is implemented as a registry of additional datatypes, which extends the possible \texttt{dtype}s recognized by TVM.
Importantly, very little compiler logic in TVM depends directly on the value of \texttt{dtype}, and so new datatypes can be incorporated with minimal modification to TVM itself.
There are a few key places where TVM must be modified, though; this is what we describe now.

There are two primary ways in which the user interacts with the datatype registry.
First is datatype registration, in which a user defines a new datatype by providing a name and a datatype ID.
The datatype name allows TVM to parse programs which use the custom datatype; 
the ID is an arbitrary unique number assigned to the type, used in TVM's internal representation.
Everywhere where TVM interacts with the \texttt{dtype} of a program node, we must now handle the chance that the \texttt{dtype} is not one of the types hard-coded into TVM.
We modify TVM to, in these cases, consult the datatype registry for the unrecognized datatype;
if the datatype is found, then TVM can proceed as normal.


\begin{figure}
    \centering
    \includegraphics[width=\linewidth]{figures/lowering.png}
    \caption{The expected result of a user's registered lowering function. A lowering function should convert a program in TVM's IR using custom datatypes (in this case \bfloat{}), to a program in TVM's IR which native TVM can understand and compile (in this case, a call to an external library, taking two \texttt{uint16\_t}s).}
    \label{fig:lowering}
\end{figure}

Once a datatype is registered, programs can be written which use the datatype.
However, once a user goes to compile the program from TVM's intermediate representation to an external language such as LLVM or CUDA, TVM will hit a wall, as it will not understand how to compile operations over custom datatypes.
This brings us to the second way in which the user interacts with the registry:
the user is expected to register \textit{lowering functions} for their datatype, which tell TVM how to lower operations over their datatype.
The user is not expected to lower operations over their datatype directly to LLVM or CUDA;
instead, the lowering functions should lower the custom datatype code into TVM code which native TVM can understand and compile.


Figure \ref{fig:lowering} illustrates what we mean.
Let's assume we are a datatype researcher interested in exploring the \bfloat{} type, and have chosen to run some workloads by plugging a \bfloat{} library that we have built into TVM via the Bring Your Own Datatypes framework.
In this case, our workload is represented as a simple program, showing an Add node with type \bfloat{}, and two \bfloat{} inputs.
Native TVM does not understand how to compile this code to, for example, LLVM---but we do!
We have a library implementing our datatype, presumably containing an implementation of \bfloat{} add at the very least.
Thus, our Add node should become a Call node, calling out to a function (call it BFloat16Add) in our library.
Finally, we can store the bits of the input \bfloat{}s inside of 16-bit unsigned integers.
The resulting program is one that TVM can understand and compile---it is simply a call to an external library function, taking two unsigned integers.

\begin{figure}
    \centering
    \begin{lstlisting}[numbers=none]
CDLL("bfloat16.so", RTLD_GLOBAL)
tvm.datatype.register("bfloat16", 129)
tvm.datatype.register_op(
    tvm.datatype.create_lower_func("BFloat16Add"),
    "Add", "llvm", "bfloat16")
    \end{lstlisting}
    \caption{Using TVM's Python frontend to load a \bfloat{} library, register \bfloat{} with the framework, and register a lowering function for \bfloat{} add.}
    \label{fig:bfloat}
\end{figure}

Figure \ref{fig:bfloat} shows an example of using the TVM Python frontend to interact with the Bring Your Own Datatypes framework in the two ways described above.
We first bring the datatype library into the process space using the \texttt{CDLL()} Python function.
Then, we register the datatype, giving it name \texttt{"bfloat16"} and type code 129.
Finally, we add a lowering function for \bfloat{} add, using the convenience function \texttt{create\_lower\_func()}, which creates the lowering function described visually in Figure \ref{fig:lowering}.
Specifically, this function creates a lowering function which, given a program node such as Add, lowers the node to a Call to a function (where the function is specified by the function name given, in this case \texttt{"BFloat16Add"}).


## References
[ieee754]: TODO
[jouppi2017datacenter]: TODO
[posittensorflow]: TODO
