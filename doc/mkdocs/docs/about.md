# About Holoflow

Holoflow is an open-source collection of libraries for building high-performance signal-processing applications. It is developed by [Jules Guillou](https://github.com/JulesGuillou) under the scientific direction of [Michael Atlan](https://www.pariseyeimaging.com/Members/180ab8642a-Michael-Atlan.en.htm), a tenured researcher at CNRS, to support digital Doppler holography experiments and real-time retinal blood-flow imaging.

The project is organized as a layered stack. [Holovibes](holovibes/index.md) is the interactive application used in the laboratory, [Holoflow](holoflow/index.md) is its graph-based execution runtime, and smaller libraries provide reusable processing tasks, numerical operations, event routing, GPU resource management, and high-throughput file I/O.

## Why develop Holoflow?

Holoflow's requirements come from the intersection of real-time high-throughput processing, complex mathematical and physical pipelines, a fast-growing research ecosystem, and highly parameterized applications.

Many domain scientists use NumPy[^numpy2020] or GPU-accelerated alternatives such as CuPy,[^cupy2017] JAX,[^jax2018] and PyTorch[^pytorch2019] because they abstract implementation details and let researchers focus on equations. However, a focused benchmark of a representative micro-batch laser Doppler holography pipeline found that Python-based implementations could not match optimized C++ and CUDA, which was approximately 74–267% faster depending on the platform.[^guillou2027]

Reaching the required throughput still required significant HPC expertise and low-level optimization. Holoflow aims to provide predictable, high-throughput scientific computing while retaining a high-level, declarative interface.

## From graphs to real-time execution

A processing pipeline is described as a graph of computational tasks rather than as manual buffer management and scheduling code. The runtime instantiates tasks, schedules GPU work, manages memory and tensor lifetimes, coordinates synchronization and data movement, and exposes metrics to the application.

Pipelines can combine acquisition, signal processing, reconstruction, analysis, and visualization while sustaining the data rates required by modern scientific imaging systems. Holoflow was developed for demanding digital holography workloads, but its execution model is intended for broader scientific computing applications.

The processing graph and laser Doppler angiography shown on the [home page](index.md) are based on the holographic processing described by Puyo et al.[^puyo2018] The example acquisition is derived from the dataset published by Atlan.[^atlan2025]

[^puyo2018]: L. Puyo, M. Paques, M. Fink, J.-A. Sahel, and M. Atlan, “[In vivo laser Doppler holography of the human retina](https://hal.sorbonne-universite.fr/hal-01875560v1),” *Biomedical Optics Express*, vol. 9, no. 9, pp. 4113–4129, 2018. [https://doi.org/10.1364/BOE.9.004113](https://doi.org/10.1364/BOE.9.004113). See also the [site-wide reference](references.md#puyo-2018).

[^atlan2025]: M. Atlan, *Doppler Holography Measurements of the Eye Fundus in a Volunteer – May 27, 2025* [Data set]. Zenodo, 2025. [https://doi.org/10.5281/zenodo.16761111](https://doi.org/10.5281/zenodo.16761111). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See also the [site-wide reference](references.md#atlan-2025).

[^numpy2020]: C. R. Harris, J. Millman, S. J. van der Walt, et al., “[Array programming with NumPy](https://doi.org/10.1038/s41586-020-2649-2),” *Nature*, vol. 585, no. 7825, pp. 357–362, 2020. See also the [site-wide reference](references.md#harris-2020).

[^cupy2017]: R. Okuta, Y. Unno, D. Nishino, S. Hido, and C. Loomis, “[CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations](https://github.com/cupy/cupy#reference),” in *Proceedings of the Workshop on Machine Learning Systems at NIPS 2017*, 2017. See also the [site-wide reference](references.md#okuta-2017).

[^jax2018]: J. Bradbury, R. Frostig, P. Hawkins, et al., “[JAX: composable transformations of Python+NumPy programs](https://github.com/jax-ml/jax#citing-jax),” software, 2018. See also the [site-wide reference](references.md#bradbury-2018).

[^pytorch2019]: A. Paszke, S. Gross, F. Massa, et al., “[PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://papers.neurips.cc/paper_files/paper/2019/hash=bdbca288fee7f92f2bfa9f7012727740-Abstract.html),” in *Advances in Neural Information Processing Systems 32*, pp. 8024–8035, 2019. See also the [site-wide reference](references.md#paszke-2019).

[^guillou2027]: J. Guillou, J. Fabrizio, E. Carlinet, and M. Atlan, “Real-Time Scientific Computing in Python: The Cost of High-Level GPU Abstractions,” unpublished manuscript, 2027. See also the [site-wide reference](references.md#guillou-2027).
