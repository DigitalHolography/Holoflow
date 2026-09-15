# About the Holoflow project

Holoflow is an open-source software project for high-performance signal processing and digital holography. It is developed by [Jules Guillou](https://github.com/JulesGuillou) under the scientific direction of [Michael Atlan](https://www.pariseyeimaging.com/Members/180ab8642a-Michael-Atlan.en.htm), a tenured researcher at CNRS, to support digital Doppler holography experiments and real-time retinal blood-flow imaging.

The project is organized as a layered stack. [Holovibes](holovibes/index.md) is the interactive application used in the laboratory, [Holoflow](holoflow/index.md) is its graph-based execution runtime, and smaller libraries provide reusable processing tasks, numerical operations, event routing, GPU resource management, and high-throughput file I/O.

The processing graph and laser Doppler angiography shown on the [home page](index.md) are based on the holographic processing described by Puyo et al.[^puyo2018] The example acquisition is derived from the dataset published by Atlan.[^atlan2025]

[^puyo2018]: L. Puyo, M. Paques, M. Fink, J.-A. Sahel, and M. Atlan, “[In vivo laser Doppler holography of the human retina](https://hal.sorbonne-universite.fr/hal-01875560v1),” *Biomedical Optics Express*, vol. 9, no. 9, pp. 4113–4129, 2018. [https://doi.org/10.1364/BOE.9.004113](https://doi.org/10.1364/BOE.9.004113). See also the [site-wide reference](references.md#puyo-2018).

[^atlan2025]: M. Atlan, *Doppler Holography Measurements of the Eye Fundus in a Volunteer – May 27, 2025* [Data set]. Zenodo, 2025. [https://doi.org/10.5281/zenodo.16761111](https://doi.org/10.5281/zenodo.16761111). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See also the [site-wide reference](references.md#atlan-2025).
