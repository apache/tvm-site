---
layout: home
---

{::options parse_block_html="true" /}


<a name="about"></a>

<section class="aboutSec">
<div class="container">
{:.aboutInner}
* {:.aboutImgCol}
    ![Aboutimage](/assets/images/about-image.svg "aboutimage"){:.desktopImg}
    ![responsiveAbout](/assets/images/about-responsive-image.svg "responsiveAbout"){:.responsiveImg}
* {:.aboutDetailsCol}
#### About Apache TVM
Apache TVM is an open deep learning compiler stack for CPUs, GPUs, and specialized accelerators. It aims to close the gap between the productivity-focused deep learning frameworks,
and the performance- or efficiency-oriented hardware backends. TVM provides the following main features:

    * Compilation of deep learning models into minimum deployable modules.
    * Infrastructure to automatic generate and optimize models on more backend with better performance.
</div>
</section>

<section class="keyfeatures">
<div class="key-section-container">
{:.key-title-text}
##  Key Features & Capabilities

* {:.key-block}
![Performance](/assets/images/speed.svg "speed")
### Performance

    {:.mb-3}
    Compilation and minimal runtimes commonly unlock  ML workloads on existing hardware.

* {:.key-block}
![Run Everywhere](/assets/images/run.svg "run")
### Run Everywhere
    CPUs,  GPUs, browsers, microcontrollers, FPGAs.

    {:.mt-0.mt-lg-3}
    Automatically generate and optimize tensor operators on more backends.

* {:.key-block}
![Flexibility](/assets/images/Flexibility.svg "Flexibility")
### Flexibility
    Need support for block sparsity, quantization (1,2,4,8 bit integers, posit), random forests/classical ML, memory planning, MISRA-C compatibility, Python prototyping or all of the above?

    {:.mt-0.mt-lg-3}
    TVM’s flexible design enables all of these things and more.

* {:.key-block}
![Ease of Use](/assets/images/use.svg "Ease of Use")
### Ease of Use
Compilation of deep learning models in Keras, MXNet, PyTorch, Tensorflow, CoreML, DarkNet. Start using TVM with Python today, build out production stacks using C++, Rust, or Java the next day.
</div>

</section>

<section class="organizationsSec">
<div class="container">

### Organizations using and contributing to TVM
Apache TVM is actively being used and contributed to by developers from many organizations.
This is a community maintained list of organizations using and contributing to tvm in alphabetical order.
{% include contrib_orgs.html %}
</div>
</section>


<a name="community"></a>

<section class="communitySec">
<div class="container">
### Join the TVM Community
Here are a few ways you can stay involved:
{% include community.html %}
</div>
</section>




<section class="docSec">
<div class="container">

* {:.doc-link-block}
### Docs
[Written with care <br/> & love for you.](https://tvm.apache.org/docs/)

* {:.doc-link-block}
### Github
[Github repo <br/> for development](https://github.com/apache/incubator-tvm)

* {:.doc-link-block}
### Blog
[Read more about TVM <br/> and our thinking](/blog)

</div>
</section>
