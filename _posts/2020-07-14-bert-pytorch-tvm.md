---
layout: post
title: "Bridging PyTorch and TVM"
author: "Thomas Viehmann, MathInf GmbH"
date: 2020-07-14
---
{% include JB/setup %}

(A more code-heavy variant is crossposted on the more PyTorch affine [Lernapparat](https://lernapparat.de/transformers-pytorch-tvm/),
 the Jupyter Notebook to follow along is on [github](https://github.com/t-vi/pytorch-tvmisc/tree/master/transformers-pytorch-tvm/).)

Some of the most intriguing applications of Artificial Intelligence have been in Natural Language Processing.
Models like BERT or GPT-2 and their variants can seemingly grasp enough of a text to continue it in a way that needs a second look to recognize as gibberish.

These models belong to a class of neural network architectures called *Transformers*. One of the favourite libraries
implementing them is the [HuggingFace transformers library](https://github.com/huggingface/transformers/).

But, in contrast to convolutional models or LSTMs where we have heavily optimized implementations, this is not as much the case for transformers.
So here we explore how TVM can fill the gap. We will do so in two steps:

- First we look at BERT inference and tuning that on TVM.
- Secondly, we start some more fundamental exploration of how one could use TVM for training in PyTorch.
  Given the experimental nature, we focus on feasibility more than on the performance in this part.

# Optimizing BERT Inference with TVM

So how do we get BERT from the transformer library to TVM?

Helpfully, transformers supports tracing their model with the PyTorch JIT. We use their [tutorial on it](https://huggingface.co/transformers/torchscript.html),
specifically the part until we have a traced model.

The PyTorch traced model takes around 0.65-0.7 seconds for 100 runs on my AMD Radeon VII with the example inputs, which means 6.5-7ms per run.
We can try to see if we can use TVM get faster. Let converting our model to TVM is a breeze:


```python
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]

mod_bert, params_bert = tvm.relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")
```

There will be a few warnings about not finding dtype information, but it goes well!
We can now build and run it. Building follows the standard TVM recipe. We also convert the PyTorch (cpu) tensors to TVM arrays.


```python
target = 'rocm -model=gfx906'  # use what matches your GPU

target_host = 'llvm'
ctx = tvm.context(target)

tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx)
st_a = tvm.nd.array(segments_tensors.numpy(), ctx)
```


```python
tvm.relay.backend.compile_engine.get().clear() # just to be sure, see https://github.com/apache/incubator-tvm/pull/5724

with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = tvm.relay.build(mod_bert,
                                     target=target,
                                     target_host=target_host,
                                     params=params_bert)
module = tvm.contrib.graph_runtime.create(graph, lib, ctx)
```

This will warn us a few times times:
```
    WARNING:autotvm:Cannot find config for ... batch_matmul.cuda .... A fallback configuration is used, which may bring great performance regression.
```

Uh oh, _may bring great performance regression_. Let us see.

But first we run the model and see if the outputs match:


```python
    (8.583069e-06, 8.493662e-07)
```

Looks good. Remember that we're computing in float32, so $10^{-6}$ish is a good result.

After building our model and setting the parameters, we time our model like this:

```python
def x():
    for i in range(100):
        module.run()
    ctx.sync()
x()
%timeit x()
```

Ouch, it takes 6.65s per 100 runs, or 67ms per run of the model. That's slow indeed. But the warning said that is was because it could not find (tuned) configurations. Let us then tune the tasks.

Tuning does take half a day or so (I'm basically following the TVM tuning tutorial for ResNet tuning with autotvm.)

After this, we can again build the model, this time with the new configuration. This time we should see no comments about missing configurations.
Now it's in the region of 6.5-7ms per run, similar to PyTorch. This is what we get from this very elementary optimization of our operators. We can push it a little further, though.

To see how, let us dive deep into BERT modeling and TVM.

If you don't want to get the full details, do skip the next section and scroll down to _Results_. I should add that I would hope that this tuning part of the tutorial will obsolete itself in the sense that in some near future, you will get much better speed right out of the box or at least after some initial tuning. So if you don't see a speedup between here and _Results_, that's because I did my homework in submitting patches.

## The BERT model

Let us take a closer look at what's going on in BERT.

Like many deep learning models, BERT comes with a bit some prologue (vocabulary embeddings) and epilogue (pooling) and the bulk is organized into similar-looking blocks, here we have 12 `BertLayer` modules.
The `attention_mask` is jsut to prevent BERT from looking at the answer when dealing with the question.

![Bert Model](/images/bert-pytorch/bert_model.svg){: width="100%" }

So let us zoom in and look at a BertLayer in detail, since that ultimately is what we need make fast.
As we see in the net diagram, the main part of the `BertLayer` module is a submodule `BertSelfAttention`.

![BertLayer](/images/bert-pytorch/bert_layer.svg){: width="100%" }

Now the `BertSelfAttention` captures the famed self-attention mechanism that is the hallmark of transformer models. (I cannot recommend Sascha Rush's [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) enough as a detailed walkthrough.)

## Putting the BertLayer under the Microscope

If we want go into details, we should want to run a BertLayer individually.
We grab the inputs of a BertLayer (see the Notebook for how) and convert a single `BertLayer` to TVM as we did for the entire model.

To look at the TVM module, we define a little visualization helper (loosely based on TVM [PR#4370](https://github.com/apache/incubator-tvm/pull/4370)).


```python
import graphviz
def visualize(expr, collapse_small=True, node_attr_dict = {}):
    def collect_ops(node):
        ops = set()
        def visitor(e):
            if isinstance(e, tvm.ir.Op):
                ops.add(e.name)
        tvm.relay.analysis.post_order_visit(node, visitor)
        return ops

    # node_dict maps a Relay node to an index (node ID)
    def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        node_dict[node] = len(node_dict)

    node_dict = {}
    tvm.relay.analysis.post_order_visit(expr, lambda x: _traverse_expr(x, node_dict))

    relayviz_nodes = []

    dot = graphviz.Digraph(format='svg', )
    dot.attr('node', shape = 'box')

    def to_str(node):
        if isinstance(node, tvm.relay.Constant):
            return repr(node).lstrip('Constant(')[:-1]
        else:
            raise NotImplementedError("to_str:" + repr(node))

    def is_small_const(c):
        if not (collapse_small and isinstance(c, tvm.relay.Constant)):
            return False
        if isinstance(c.data, tvm.runtime.ndarray.NDArray):
            return numpy.prod(c.data.shape) < 10
        return True

    # Sort by node ID
    for node, node_id in sorted(node_dict.items(), key=lambda x: x[1]):
        if isinstance(node, tvm.relay.Function):
            dot.node(str(node_id), 'Function', **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.body]), str(node_id))
        elif isinstance(node, tvm.relay.Var):
            if node.type_annotation is not None:
                if hasattr(node.type_annotation, 'shape'):
                    shape = tuple([int(x) for x in node.type_annotation.shape])
                    dtype = node.type_annotation.dtype
                    typstr = 'Tensor[{}, {}]'.format(shape, dtype)
                else:
                    typstr = str(node.type_annotation)
            else:
                typstr = '?'
            d = dict(shape = 'ellipse')
            d.update(node_attr_dict.get(node, {}))
            dot.node(str(node_id),
                     '{}: {}'.format(
                         node.name_hint, typstr
                     ), **d)
        elif isinstance(node, tvm.relay.Tuple):
            dot.node(str(node_id), 'Tuple[...])', **node_attr_dict.get(node, {}))
            for field in node.fields:
                dot.edge(str(node_dict[field]), str(node_id))
        elif isinstance(node, tvm.relay.Constant):

            if not is_small_const(node): # small consts are shown in ops
                dot.node(str(node_id), 'Constant({}, {})'.format(node.data.shape, node.data.dtype),
                        **node_attr_dict.get(node, {}))
        elif isinstance(node, tvm.relay.Call):
            args_with_edge = []
            arg_str_list = []
            for arg in node.args:
                if is_small_const(arg):
                    arg_str_list.append(to_str(arg))
                else:
                    arg_str_list.append('·')
                    args_with_edge.append(arg)
            arg_str = ', '.join(arg_str_list)
            if isinstance(node.op, tvm.ir.Op):
                name = node.op.name
                attrs = {k:getattr(node.attrs, k) for k in node.attrs.keys()} if hasattr(node.attrs, 'keys') else {}
                #attrs = inspect.getmembers(node.attrs)
                attr_str_list = [k+'='+(str(v) if len(str(v))<20 else "...") for k, v in attrs.items()]
                if attr_str_list:
                    attr_str = '| '+ ', '.join(attr_str_list)
                else:
                    attr_str = ''
            else:
                ops = collect_ops(node)
                if ops:
                    name = '_'.join(ops)
                else:
                    name = '...'
                attr_str = ''
            s = f'{name}({arg_str}{attr_str})'
            dot.node(str(node_id), s, **node_attr_dict.get(node, {}))
            for arg in args_with_edge:
                dot.edge(str(node_dict[arg]), str(node_id))
        elif isinstance(node, tvm.ir.Op):
            # dot.node(str(node_id), 'Op {}'.format(node.name))
            pass # covered in call
        elif isinstance(node, tvm.relay.TupleGetItem):
            dot.node(str(node_id), 'TupleGetItem(idx={})'.format(node.index), **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.tuple_value]), str(node_id))
        elif isinstance(node, tvm.relay.Let):
            dot.node(str(node_id), 'Let(XX)', **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.value]), str(node_id))
            dot.edge(str(node_id), str(node_dict[node.var]))
        else:
            raise RuntimeError(
                'Unknown node type. node_id: {}, node: {}'.format(node_id, type(node)))

    return dot

```

Let's run that on our main function. For some reason (well, to be fully general, probably) the PyTorch converter will convert `Linear` layers to `batch_matmul` rather than just `dense`. We'll get back to this in a bit. As TVM's `batch_matmul` has the contraction axis last on both operands (unlike PyTorch), there are quite a few transpose operations, too.


```python
visualize(mod['main'])
```

![svg](/images/bert-pytorch/bert-tvm_49_0.svg){: width="100%" }


In addition to our named inputs, we see a number of unnamed (numbered) variables. These are the neural network parameters.

Let us compile our model.

Just like the full model, we can run and time our submodule after checking that it computes the same quantities.

100 runs take 20.2ms. The back of the envelope calculation here is that with `BertLayer` in PyTorch we are spending about 0.2ms in this layer, so about 2.4ms on 12 layers - a not the majority but a sizeable part of the 6-7ms overall runtime. Let's compare to TVM. (A good rule is to never optimize without measuring.)

Similarly, TVM clocks in at 18.2ms for 100 runs. So here we are again roughly on par with PyTorch.

One thing we see from the picture is that the input is reshaped three times. There is a TVM optimization pass call Common Subexpression Elimination (CSE) that combines the three reshapes.
(A while ago, this did not succeed because it had distinct shape arguments, but this was since solved by the TVM developers in the dynamic to static conversion pass.)
Also, the model parameters that are reshaped and transposed. Can we get rid of that, too?
Yes. And for that we would first _bind_ the parameters, i.e. put them into the model. Then the parameters have become constants instead of input nodes.
With the `Foldconstant` pass, we can propagate the constants through the `transpose`s and `reshape`s to move them closer to the matmuls.

After these three (which TVM will do when we compile a relay model), our model looks like this:

![svg](/images/bert-pytorch/bert-tvm_72_0.svg){: width="100%" }

And now comes an interesting trick. It is more efficient to merge the three batch matmuls with the same input into a single `batch_matmul`. We implemented a pass doing this in [TVM PR 5791](https://github.com/apache/incubator-tvm/pull/5791). So let's call it and also have another constant-folding pass.


```python
new_mod = tvm.relay.transform.CombineParallelBatchMatmul()(new_mod)
new_mod = tvm.relay.transform.FoldConstant()(new_mod)
visualize(new_mod["main"])
```

![svg](/images/bert-pytorch/bert-tvm_74_0.svg){: width="100%" }

Awesome. After checking that we still get the same result.
We can time again: 25.2 ms for 100 runs. It's a bit slow again because we need to tune for the new shapes.
After tuning, we are at 12.6ms for 100 runs, so we went from about 0.2ms to about 0.13-0.15ms, a nice speedup.
By our handwavy calculation, this should cut 0.6-0.8ms from the total runtime, or somewhere between 5%-10%. Let's check.

## Results on the overall BERT model after optimization

Let's define a function combining the optimization passes from above and run it on the entire BERT model.
We go through the same exercise as above.

We get to 624ms for 100 runs. So yay, we went from 6.5-7ms in PyTorch to ~6.2ms in TVM. This is a 5%-10% speedup. Note that we have only taking a particular, not very large shape. A more serious analysis would consider more problem shapes.

We could probably take it a bit further yet - e.g. fusing the additions after the batch matmul by handling the reshape, but we'll leave it at this for now. Also we will benefit from further improvements to TVM, so it will be interesting to see how the benchmark improves over time. In particular, the upcoming Ansor tuning mechanism seems promising.

## A peek under the hood

### Comparing implementation of models

As you can see, I have always compared PyTorch with TVM outputs to see if they're good.
Also, when I investigated some inner layer, I grabbed the inputs to that to convert and feed into the TVM model. I do believe that this is a very effective technique.

Sometimes, however, it is difficult to assess whether a deviation between the results is from numerical accuracy or from an error somewhere.
When I initially converted the model, the the `SelfAttention` submodule output was replicated by the TVM model to about 1e-6.
However, the BertLayer conversion had something like 1-e3. I was not entirely clear whether that might be due to accumulated numerical errors or some material deviation somewhere.
(This turned out to be the GELU activation, which was converted to FastGELU.)

One of the things I like to do in this case is jump to double precision and check there. Numerical errors should get much smaller, while other deviations would remain of the same order.
With the PyTorch frontend, you can trace the model converted to float64 on the PyTorch side if you pass `default_dtype="float64"` to the conversion function.

Running the module and comparing to PyTorch should now have 1e-14 or so deviation.

### Improvements in TVM to facilitate this usecase

Before this worked as shown here, we had to close some gaps (but a recent git checkout will include all of them):
- The TVM PyTorch converter did not support inputs other than fp32. We [implemented improved conversion](https://github.com/t-vi/tvm/tree/pytorch_frontend_type_fix), now also included in TVM upsteam.
- The TVM schedule, i.e. the organization of the computation, of the workhorse operation, batch_matmul, was fixed and it was very slow (similar to running without a tuned schedule now). So we [implemented a tuneable schedule](https://github.com/apache/incubator-tvm/pull/5752).
- The PyTorch converter produces batch matmul operations (it could probably also be changed to produce dense layers instead). But as we saw, one of the larger speed advantages is to combine Query Key and Value linear layers, so we implemented [fusing batch matmul operations](https://github.com/apache/incubator-tvm/pull/5791).
- When comparing the computation results, we noticed that the [GELU](https://pytorch.org/docs/master/generated/torch.nn.GELU.html) function was converted to its FastGELU variant. We fixed that. (There is a _fast math_ optimization pass in TVM that does some replacement of the error function, though we didn't check if it yields FastGELU for the GELU expressed with the error function.)
- TVM was initially (and still is to a some extent) focussed on static shapes. Recently it experiments with dynamic operations. The dynamic reshape - taking an argument for the target shape - is an early of these experiments, but as seen above, it prevented the fusion of batch matmuls because the common subexpression elimination pass didn't detect that it could merge the identical input reshaping. This has improved recently.

# Training Pytorch models with TVM computation

In this second part we want see if we could use TVM while training BERT in PyTorch.
Of course, this opens an entire new can of worms as we need to deal with autodifferentiation.
While we stay with the theme from above and take `BertLayer` as the example, our methodology is representative of non-trivial modules in general.
We will want to divert the computation during training to TVM.

So the user can take a (traceable) module and do
```
add_tvm_dispatch(module, sample_input)
```
and then if she calls module with inputs of the same shape as the sample_input, she'll get the outputs computed by TVM (as PyTorch tensors, of course) and if not, it'll just use the regular forward.

The but so we already hinted at the bad news: In this part we will see how to do these things. We will not yet achieve a great speedup.

But enough talk, let us dive right in!
Again, we get our relay model with running a traced `BertLayer` from the transformer `Bert` model through `tvm.relay.frontend.from_pytorch`.

One thing we'll do in between is to move from a modular interface in PyTorch - with named parameters - to a functional
interface (which is what TVM can do for us). The first thing we want to do for that is arrange for the function arguments to be in an order that we can work with - i.e. first the direct inputs to the module and then the parameters in the same order that PyTorch uses them. After this operation, our `BertLayer ` in TVM looks like this:

![svg](/images/bert-pytorch/pytorch-tvm-training_20_0.svg){: width="100%" }

As in the BERT inference, we want to run some optimization passes.

But we also have a few new transformations:

- One particularity of the Autodifferentiation is that it'll use a lot of `..._like` operations to broadcast or "unbroadcast" (summation is the dual of broadcasting w.r.t. autodifferentiation) things. But this means that you now have two tensor arguments, even if the latter doesn't really need a gradient. `ZappLike` replaces those operations with the corresponding functions taking a shape parameter instead.
- Another thing is the "rooting" of derivatives. TVM generates a tensors with all ones of the same shape as the return values of our function as the starting point for the chain rule. These are then multiplied to the derivatives of our operations. But multiplication with ones is not doing much, so we strike that. Similarly, TVM initializes the gradient of a variable (an input) to zeros of the same shape. If it isn't used, the gradient will be zero, but if it is, the "real gradient" will be added to that zero. But adding zero can be eliminated as well. These are taken care off by ZeroZapp and OneZapp.
- TVM doesn't have a training variant for the `LayerNorm` (or `BatchNorm` or others). So we implement a pass to spell out the computation.
- TVM also doesn't have training dropout. Here the problem is somewhat harder to fix, as TVM doesn't have random currently. We instead replace the dropout by a construct taking a random bernoulli draw (of 0/1 values) and mimicking dropout with that. The idea is that we'll use PyTorch to generate this mask for us. This has the added benefit that (if we generate dropout masks in the same order as PyTorch) we'll get the exact same result.

As hinted at above, TVM's gradient taking assumes that it is the last element in the computation (the ones-Tensors discussed above). This isn't a good fit with PyTorch's modular view which expects a `grad_out` for each output to be given. Happily, this is computationally equivalent to multiplying by grad out and summation, so we amend our function with that. We wish to be flexible, so we allow both functions returning a single tensor and those returning a tuple of tensors.

With these modificaitons applied, our model looks like this:

![svg](/images/bert-pytorch/pytorch-tvm-training_25_0.svg){: width="100%" }

Finally we can take the grad. As we get a lot of `let` nodes, we bring it to normal form using the `ToGraphNormalForm` pass.
TVM's gradient-taking returns a function that has the same parameters as the original function (in our case amended with the `grad_out` and dropout) and then returns a tuple of the original return and a tuple containing gradients for all inputs.
The first thing we do is to drop all the gradients for `grad_out` and `dropout` which we don't need.
Then we run our simplification passes.

So this is the graph we have now for forward and backward:

![svg](/images/bert-pytorch/pytorch-tvm-training_31_0.svg){: width="100%" }

But in PyTorch, we first compute the forward and then the backwards, so we have to take out the saw and
split our graph. One of the difficult problems is what to do with things computed for both forward and backward. It is a hard problem, related to the MinCut problem.

Our extremal options could be:
- One could only keep the inputs and recompute everything as needed.
- If we had a salar output, we could compute the gradient and multiply with the derivative of the later layers on backward. (Loss functions might do that.) This does not, however, work for non-scalar tensor outputs.

We'll do the following: We compute the forward normally, but we keep all things that will be used in the backward. This is too much, unfortunately, and it is very likely the reason we don't see an end to end speedup. We'll discuss some potential heuristics below.

We use a coloring here. First we color all nodes of the forward computation in red. Then we traverse the gradient calculation and then color the nodes it needs from the backward blue. This gives us a chance to show off the attribute support in our visualization.

A bit of (PyTorch) terminology: When we have a function *Layer : x ↦ y* followed by some *Loss: y ↦ l ∈ ℝ*, the backward is *BackwardOfLayer : grad`_`out ↦ grad`_`in* with *grad`_`out = dl/dy* and *grad`_`in = dl/dx`.

![svg](/images/bert-pytorch/pytorch-tvm-training_34_0.svg){: width="100%" }

In order to split the function as described above, we collect the blue nodes as to capture - but constants will
just be duplicated and inputs (`Var` nodes) need to be treated separately.
Now we can split out the backward, replacing all the blue nodes with variables.

Next we take the forward and amend it to also return the required intermediates. The forward then looks like this:

![svg](/images/bert-pytorch/pytorch-tvm-training_40_0.svg){: width="100%" }

TVM cannot return nested tuples, so we flatten the output in the function. Again we differentiate between tensor-valued functions and tuple valued ones (i.e. those returning potentially multiple tensors).

And at last, we can let TVM do its magic and compile our functions, say to `gr_only_compiled_module`
and `fw_and_cap_compiled_module`.
Time to give it a spin. We define convenience functions to move tensors between PyTorch and TVM and get the model parameters as a TVM dictionary.


```python
def tensor_to_tvm(t):
    return tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(t))
def tensor_from_tvm(a):
    return(torch.utils.dlpack.from_dlpack(a.to_dlpack()))

model_params_tvm = {k: tensor_to_tvm(v) for k, v in pytorch_model.state_dict().items()}
```

Similarly, we get the inputs on the GPU in PyTorch and TVM.

We need to deal with the dropout. It will turn out that our record of the three dropout random draws happens in the same order as the dropout in the model. We did a depth-first search on the computational graph to find them and if the values of the the dropout are connected in the graph rather than being on independent branches, this will be the order in which PyTorch draws the matrices, too. If not, good luck fiddeling with the order.

```python
torch.manual_seed(12345)
drop_c = {}
for k in dropout_info.keys(): # we don't know the order
    p, typ = dropout_info[k]
    drop_c[k] = torch.nn.functional.dropout(torch.ones([int(i) for i in typ.shape],
                                              dtype=getattr(torch, typ.dtype), device="cuda"), p=p)*(1-p)

drop_tvm = {n: tensor_to_tvm(t) for n, t in drop_c.items()}
```

Now we can run the forward.

```python
fw_and_cap_compiled_module.set_input('input', inp_tvm[0])
fw_and_cap_compiled_module.set_input('attention_mask', inp_tvm[1])
fw_and_cap_compiled_module.set_input(**model_params_tvm)
fw_and_cap_compiled_module.set_input(**drop_tvm)
fw_and_cap_compiled_module.run()
```

And we can compare the output to PyTorch's:

```python
torch.manual_seed(12345)
pytorch_model.train()
res = pytorch_model(*inp_c)[0]
numpy.abs(fw_and_cap_compiled_module.get_output(0).asnumpy()-res.detach().cpu().numpy()).max()
```

This gives `2.1457672e-06`.

Supergood. Let's also try the backward. We generate a `grad_out`, set all the variables and run the backward model and run the backward model


```python
gr_out_c = torch.randn(res.shape, device="cuda", dtype=res.dtype)
```

```python
num_captures = len(capture_vars)
num_regular_outputs = len(fw_and_cap_fn_flattened.body.fields) - num_captures
captured_values = {v.name_hint: fw_and_cap_compiled_module.get_output(num_regular_outputs + i) for i, v in enumerate(capture_vars)}

gr_only_compiled_module.set_input(**drop_tvm)
gr_only_compiled_module.set_input(**model_params_tvm)
gr_only_compiled_module.set_input(**captured_values)
gr_only_compiled_module.set_input('gr:out:0', tensor_to_tvm(gr_out_c))
gr_only_compiled_module.run()
```

On the PyTorch side, it is easiest to re-run the forward (remembering to reset the random seed) and get the grads.


```python
torch.manual_seed(12345)
pytorch_model.train()
inp_c_rq = [i.requires_grad_() for i in inp_c]
for p in pytorch_model.parameters():
    p.requires_grad_()
res = pytorch_model(*inp_c_rq)[0]
grads_pt = torch.autograd.grad(res, inp_c_rq + list(pytorch_model.parameters()), gr_out_c, allow_unused=True)

```

Did it work? It seems so:


```python
for i, g_pt in enumerate(grads_pt):
    print(numpy.abs(gr_only_compiled_module.get_output(i).asnumpy() - g_pt.cpu().numpy()).max())
```

gives us a list of numbers in the 1e-5ish range.

But we wanted to get something running in PyTorch, right?

Keeping with how PyTorch works, we first define an `autograd.Function` that the things we just did manually:

In the `forward`:

- Generate the dropout random values,
- Run the forward,
- Record the captures, inputs, and dropout values needed for backward.

In the `backward`, run the backward and return the result (as PyTorch tensors).

With that, we get a PyTorch autograd.Function calling into TVM (we would want a small wrapper for that.

Now all we need to do to achive our goal of getting a method `add_tvm_dispatch(module, sample_inputs)` is
to trace the module, create the TVM-based autograd function from it and then replace the forward that calls
that (with the parameters) if applicable or falls back to the usual forward.
Python's unlimited dynamism makes that kind of hackery relatively easy.
As all this it is not really TVM-related, we are sparing us that here (but you could check the
[companion post](https://lernapparat.de/transformers-pytorch-tvm/).

## Performance

As I said in the beginning, we aren't quite where we want to eventually be in terms of performance.
After tuning the tasks (and on the not very realistic inference example from the HuggingFace BERT + PyTorch JIT tutorial)
we run 100 iterations of the TVM-enabled BertLayer forward and backward similar to how we did it for the inference.
One iteration takes 6.2ms going through TVM versus 1.3ms on PyTorch.

So ran our model through TVM all right. But it's not as fast as the usual method yet. Here is to opportunity!

More seriously, we have two immediate paths to improve performance:

- Find a better set of captured nodes.
- Find optimizations on the TVM graph.

In terms of heuristics for the former (remember that it quite likely NP hard, i.e. I believe it is, but I didn't work out a formal proof),
one would want to re-do cheap computation, most prominently point-wise computation (or maybe anything but matmul?). But that is for another day.

I hope you enjoyed the tutorial, I look forward to your comments at <tv@lernapparat.de>.

# Acknowledgements

I had many interesting discussions with HugingFace people and Morgan Funtowicz in particular. Also the TVM contributors had many good comments during the review of the patches TVM and on the forums. The creation of this tutorial was sponsored by AMD.

# Author

[Thomas Viehmann](https://lernapparat.de/) is the founder of [MathInf GmbH](https://mathinf.eu/), Munich, Germany, a boutique training and consultancy firm focusing on Machine Learning and PyTorch.
He is a PyTorch core developer and co-authored [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch), which currently available as [free download from the PyTorch website](https://pytorch.org/deep-learning-with-pytorch).
