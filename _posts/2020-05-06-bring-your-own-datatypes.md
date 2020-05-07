---
layout: post
title:  "Bring Your Own Datatypes: Enabling Custom Datatype Exploration in TVM"
date:   2020-05-06
author: Gus Smith
---
- TODO Update date in heading
- TODO Update filename date

In this post, we describe the Bring Your Own Datatypes framework, which enables the use of custom datatypes within TVM.

## Introduction

When designing accelerators, an important decision is how one will approximately represent real numbers in hardware.
This problem has had a longstanding, industry-standard solution: the IEEE 754 floating-point standard.[^ieee]
When trying to squeeze the most out of hardware, though, will we necessarily use IEEE 754 floats?
There are a number of reasons to believe we could build a better datatype: one which is smaller, faster, or more power efficient, and overall more optimal for our workload and the constraints of our hardware design.

Researchers have already begun experimenting with new datatypes in academic and industrial accelerator designs.
For example, Google's Tensor Processing Unit (the TPU) uses the `bfloat` type: a single-precision IEEE float which has been truncated to 16 bits, instantly reducing storage cost by half while often losing no model accuracy.[^jouppi2017datacenter]
Before researchers begin building hardware for their datatype, however, they first need to determine how their datatype will behave numerically in the workloads they care about.
This often involves first building a software-emulated version of their datatype, and then hacking the datatype directly into workloads;
even better is to integrate the datatype directly into compilers themselves.
Both routes can be tedious, with the latter route often becoming unmanageable given the size and complexity of modern compilers.
One example taken from GitHub shows someone hacking the *posit* datatype into TensorFlow.[^posittensorflow]
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
In the Bring Your Own Datatypes framework,
  "datatype" means a scalar type:
  `float32`
  or `uint8`, for example.
We do not handle more complicated data formats
  such as [block floating point](https://en.wikipedia.org/wiki/Block_floating_point)
  or Intel's [Flexpoint](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/flexpoint-numerical-innovation-underlying-intel-nervana-neural-network-processor.html).
Additionally,
  we only claim to support
  *software emulated* versions of these scalar datatypes;
  we do not explicitly support compiling and running on custom datatype hardware.


Each tensor in TVM
  is assigned a type code,
  which defines the datatype of the scalars
  within the tensor.
A number of these type codes
  have hard-coded meanings in TVM,
  mapping to common datatypes
  such as `int` and `float`.
However,
  the vast majority of type codes
  are unused.
The Bring Your Own Datatypes framework
  allows users to 
  claim these unused type codes
  and add their own new datatypes
  at runtime.

The framework is implemented as
  a registry 
  which sits alongside
  TVM's normal datatype facilities.
There are two primary ways
  in which the user interacts with
  the datatype registry:
  first, **datatype registration,**
  and second, **lowering function registration.**
These steps are akin to
  *declaration* and *implementation* of the datatype,
  respectively.
  
### Datatype Registration

To register the datatype,
  the user assigns the datatype
  a name and a type code,
  where the type code comes from
  the range of unused type codes
  available to custom datatypes.
```python
tvm.datatype.register('bfloat', 150)
```
The above code registers
  the `'bfloat'` datatype
  with type code 140.
This registration step
  allows TVM to parse programs
  which use the custom type:
```python
x = relay.var('x', shape=(3, ), dtype='float32')
y = relay.var('y', shape=(3, ), dtype='float32')
x_bfloat = relay.cast(x, dtype='custom[bfloat]16')
y_bfloat = relay.cast(y, dtype='custom[bfloat]16')
z_bfloat = x_bfloat + y_bfloat
z = relay.cast(z_bfloat, dtype='float32')
program = relay.Function([x, y], z)
print(program)

# v0.0.4
# fn (%x: Tensor[(3), float32], %y: Tensor[(3), float32]) {
#   %0 = cast(%x, dtype="custom[bfloat]16");
#   %1 = cast(%y, dtype="custom[bfloat]16");
#   %2 = add(%0, %1);
#   cast(%2, dtype="float32")
# }
```
The program above
  casts `float32` inputs `x` and `y`
  into `bfloat`s,
  adds them,
  and casts the result back to `float32`.
Once the `bfloat` type is registered,
  TVM is able to parse the special `dtype` syntax
  `custom[<typename>]`,
  where `<typename>` is the name registered for the type.
This syntax also supports the usual
  `<bits>x<lanes>` format;
  here, we use `16` to indicate that
  each `bfloat` is 16 bits wide.
(The number of lanes
  defaults to 1.)
  
### Lowering Function Registration

Though TVM can parse the above program,
  it cannot yet compile it,
  as TVM does not yet understand 
  how to compile operations 
  over the `bfloat` type.
To compile these programs,
  we register *lowering functions* for the custom datatype,
  which help TVM convert the operations
  into something it can understand and compile.

Generally, the user is not expected to 
  lower operations
  directly to LLVM or CUDA.
Instead, most code using custom datatypes
  can be lowered into code which *doesn't* use custom datatypes,
  with some simple tricks.
We can then rely on native TVM
  to understand and compile the code.

{:center: style="text-align: center"}
![A lowering function lowering an add over `bfloat`s to a library call over `uint16_t`s](/images/bring-your-own-datatypes/lowering.png){: width="50%"}
{:center}
<center>
Figure 1: The expected result of a user's registered lowering function. A lowering function should convert a program in TVM's IR using custom datatypes (in this case `bfloat`) to a program in TVM's IR which native TVM can understand and compile (in this case, a call to an external library, taking two `uint16_t`s).
</center> <p></p>

Figure 1 shows a common pattern.
Let's assume we are
  interested in exploring the `bfloat` type,
  and have chosen to run some workloads
  by plugging a `bfloat` emulation library (e.g. [biovault_bfloat16](https://github.com/biovault/biovault_bfloat16){:target="_blank"}) into TVM
  via the Bring Your Own Datatypes framework.
Our workload is a simple program
  which adds two `bfloat` inputs.
Native TVM does not understand
  how to implement `bfloat` addition---but it doesn't need to,
  as we have a library implementing our datatype!
The library contains an implementation of `bfloat` addition,
  alongside other operators such as multiplication and square root.
To implement this `bfloat` addition,
  we'd just like to call into our library.
Thus, our Add node should become a Call node,
  calling out to a function (call it `BFloat16Add`) in our library.
To store the bits of the input `bfloat`s
  inside a type that TVM understands,
  we use 16-bit unsigned integers.
The resulting program 
  is one that TVM can understand and compile---it
  is simply a call to an external library function,
  taking two unsigned integers.
  
To achieve the above lowering,
  we register a lowering function
  for `bfloat`:
```python
tvm.datatype.register_op(
    tvm.datatype.create_lower_func('BFloat16Add'),
    'Add', 'llvm', 'bfloat')
```
The above code registers
  a lowering function
  for a specific operator (Add),
  compilation target (LLVM),
  and datatype (`bfloat`).
The first argument
  is the lowering function.
This can be any function
  taking a TVM IR node
  and returning a new TVM IR node.
In our case,
  we use a helper function
  provided by the Bring Your Own Datatypes framework.
`tvm.datatype.create_lower_func('BFloat16Add')`
  creates a lowering function
  for the common pattern described above.
The resulting function
  converts the arguments of the given node
  to `uint16_t`,
  and then converts the node itself
  into a call to the given function name
  (in this case, `'BFloat16Add'`).

To implement a custom datatype,
  the user will need to register
  a lowering function for every operator
  in the workload they would like to run.
For a network like ResNet,
  this will be around 10 operators,
  including things like, Add, Div, various Casts, and Max.
In our tests,
  registering a datatype
  and all lowering functions
  takes around 40 lines of Python.

# Wrapping Up
  
The Bring Your Own Datatypes framework
  brings user-defined datatypes to TVM.
We hope this will encourage datatype researchers
  to use TVM in their research;
  similarly,
  we hope this will spark interest
  in custom datatypes
  within the deep learning community.

  
---

*Gus Smith is a PhD student at the University of Washington working at the intersection of computer architecture and programming languages. His personal website is [justg.us](https://justg.us).*

## References

[^ieee]: [754-2019 - IEEE Standard for Floating-Point Arithmetic](https://standards.ieee.org/standard/754-2019.html)
[^jouppi2017datacenter]: Jouppi, Norman P., et al. "In-datacenter performance analysis of a tensor processing unit." Proceedings of the 44th Annual International Symposium on Computer Architecture. 2017.
[^posittensorflow]: [xman/tensorflow](https://github.com/xman/tensorflow)
