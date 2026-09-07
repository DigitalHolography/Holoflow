---
template: home.html
title: Holoflow
hide:
  - navigation
  - toc
---

<section class="hf-hero" aria-labelledby="hf-hero-title">
  <div class="hf-shell hf-hero__layout">
    <div class="hf-hero__copy">
      <p class="hf-eyebrow">The Holoflow project</p>
      <h1 id="hf-hero-title">Real-time digital holography, from acquisition to insight.</h1>
      <p class="hf-lede">
        Holoflow is an open-source C++ and CUDA stack for describing, running, and exploring
        high-throughput scientific imaging pipelines. Holovibes brings the stack into the lab
        through an interactive Qt application.
      </p>
      <div class="hf-actions">
        <a class="md-button md-button--primary" href="#project">Explore the project</a>
        <a class="md-button" href="https://github.com/DigitalHolography/Holoflow">View on GitHub</a>
      </div>
      <!-- <ul class="hf-facts" aria-label="Core technologies">
        <li>C++</li>
        <li>CUDA</li>
        <li>Qt</li>
        <li>Real-time processing</li>
      </ul> -->
    </div>
    <figure class="hf-hero__visual">
      <video autoplay loop muted playsinline preload="metadata" aria-label="Processed laser Doppler angiography">
        <source src="assets/videos/DEMO_PROCESSED_NA_20260904_153832_AQ002_60fps_512x512.mp4" type="video/mp4">
      </video>
      <figcaption>Laser Doppler angiography processed in real time.</figcaption>
    </figure>
  </div>
</section>

<section class="hf-section hf-section--tinted" aria-labelledby="showcase-title">
  <div class="hf-shell">
    <header class="hf-section__heading">
      <p class="hf-eyebrow">Experiments and engineering</p>
      <h2 id="showcase-title">What we build with it</h2>
      <p>One stack connects the processing model, the GPU runtime, and the measurements seen in the lab.</p>
    </header>

    <article class="hf-feature">
      <div class="hf-feature__media hf-feature__media--diagram">
        <img
          src="assets/images/holoflow-pipeline.svg"
          alt="Holoflow graph connecting acquisition, GPU processing tasks, and display output"
          loading="lazy"
        >
      </div>
      <div class="hf-feature__copy">
        <h3>A graph backend for scientific pipelines</h3>
        <p>
          Describe the computation as connected tasks. Holoflow compiles the graph, manages tensor
          lifetimes and memory, then schedules CPU and GPU work while the algorithm remains readable.
        </p>
        <a class="hf-text-link" href="holoflow/index.html">Understand the runtime <span aria-hidden="true">→</span></a>
      </div>
    </article>

    <article class="hf-feature hf-feature--experiment hf-feature--reverse">
      <div class="hf-feature__media">
        <div class="hf-video-pair">
          <figure>
            <video autoplay loop muted playsinline preload="metadata" aria-label="Raw interferometric frames">
              <source src="assets/videos/DEMO_RAW_NA_20260904_153832_AQ004_60fps_512x512.mp4" type="video/mp4">
            </video>
            <figcaption>Interferometric input</figcaption>
          </figure>

          <figure>
            <video autoplay loop muted playsinline preload="metadata" aria-label="Processed laser Doppler angiography">
              <source src="assets/videos/DEMO_PROCESSED_NA_20260904_153832_AQ002_60fps_512x512.mp4" type="video/mp4">
            </video>
            <figcaption>Doppler angiography</figcaption>
          </figure>
        </div>
      </div>

      <div class="hf-feature__copy">
        <h3>Laser Doppler holography</h3>
        <p>
          Transform high-rate interferometric acquisitions into live views of retinal blood flow,
          with reconstruction and visualization performed as the data arrives.
        </p>
        <a class="hf-text-link" href="applications/doppler-holography.html">
          Reproduce the experiment <span aria-hidden="true">→</span>
        </a>
      </div>
    </article>


    <article class="hf-feature hf-feature--experiment">
      <div class="hf-feature__media hf-feature__media--placeholder">
        <div class="hf-video-pair">
          <figure>
            <video autoplay loop muted playsinline preload="metadata" aria-label="Temporary Doppler OCT input">
              <source src="assets/videos/DEMO_RAW_NA_20260904_153832_AQ004_60fps_512x512.mp4" type="video/mp4">
            </video>
            <figcaption>Interferometric input</figcaption>
          </figure>

          <figure>
            <video autoplay loop muted playsinline preload="metadata" aria-label="Temporary Doppler OCT output">
              <source src="assets/videos/DEMO_PROCESSED_NA_20260904_153832_AQ002_60fps_512x512.mp4" type="video/mp4">
            </video>
            <figcaption>Doppler OCT</figcaption>
          </figure>
        </div>

        <span class="hf-placeholder-label">OCT footage coming soon</span>
      </div>

      <div class="hf-feature__copy">
        <h3>Doppler OCT</h3>
        <p>
          Use the same modular processing foundation to explore depth-resolved Doppler measurements
          and evolve experimental pipelines without rebuilding the application around each method.
        </p>
        <a class="hf-text-link" href="applications/doppler-oct.html">
          Open the application guide <span aria-hidden="true">→</span>
        </a>
      </div>
    </article>


    <article class="hf-feature hf-feature--experiment hf-feature--reverse">
      <div class="hf-feature__media hf-feature__media--placeholder">
        <div class="hf-video-pair">
          <figure>
            <video autoplay loop muted playsinline preload="metadata" aria-label="Temporary wavefront analysis input">
              <source src="assets/videos/DEMO_RAW_NA_20260904_153832_AQ004_60fps_512x512.mp4" type="video/mp4">
            </video>
            <figcaption>Interferometric input</figcaption>
          </figure>

          <figure>
            <video autoplay loop muted playsinline preload="metadata" aria-label="Temporary wavefront analysis output">
              <source src="assets/videos/DEMO_PROCESSED_NA_20260904_153832_AQ002_60fps_512x512.mp4" type="video/mp4">
            </video>
            <figcaption>Wavefront analysis</figcaption>
          </figure>
        </div>

        <span class="hf-placeholder-label">Wavefront footage coming soon</span>
      </div>

      <div class="hf-feature__copy">
        <h3>Wavefront analysis</h3>
        <p>
          Inspect wavefront quality as the experiment runs, including Shack–Hartmann measurements
          and Zernike modes used to understand and correct optical aberrations.
        </p>
        <a class="hf-text-link" href="applications/wavefront-analysis.html">
          Open the application guide <span aria-hidden="true">→</span>
        </a>
      </div>
    </article>

    <article class="hf-feature hf-feature--experiment">
      <div class="hf-feature__media hf-feature__media--placeholder">
        <div class="hf-video-pair">
          <figure>
            <video autoplay loop muted playsinline preload="metadata" aria-label="Temporary tomographic diffractive microscopy input">
              <source src="assets/videos/DEMO_RAW_NA_20260904_153832_AQ004_60fps_512x512.mp4" type="video/mp4">
            </video>
            <figcaption>Interferometric input</figcaption>
          </figure>

          <figure>
            <video autoplay loop muted playsinline preload="metadata" aria-label="Temporary tomographic diffractive microscopy output">
              <source src="assets/videos/DEMO_PROCESSED_NA_20260904_153832_AQ002_60fps_512x512.mp4" type="video/mp4">
            </video>
            <figcaption>Tomographic diffractive microscopy</figcaption>
          </figure>
        </div>

        <span class="hf-placeholder-label">TDM footage coming soon</span>
      </div>

      <div class="hf-feature__copy">
        <h3>Tomographic diffractive microscopy</h3>
        <p>
          Combine off-axis holography, aberration correction, and phase unwrapping to reconstruct three-dimensional
          sample volumes from tomographic acquisitions.
        </p>
        <a class="hf-text-link" href="applications/tomographic-diffractive-microscopy.html">
          Open the application guide <span aria-hidden="true">→</span>
        </a>
      </div>
    </article>
  </div>
</section>

<section id="project" class="hf-section" aria-labelledby="project-title">
  <div class="hf-shell">
    <header class="hf-section__heading">
      <p class="hf-eyebrow">One project, several layers</p>
      <h2 id="project-title">Choose the level you need</h2>
      <p>Start with the application or work directly with the runtime and its infrastructure libraries.</p>
    </header>

    <div class="hf-projects hf-projects--primary">
      <a class="hf-project-card" href="holovibes/index.html">
        <span class="hf-project-card__kind">Desktop application</span>
        <h3>Holovibes</h3>
        <p>Acquire, reconstruct, analyze, and visualize holographic data interactively.</p>
        <span class="hf-project-card__link">Explore Holovibes <span aria-hidden="true">→</span></span>
      </a>
      <a class="hf-project-card" href="holoflow/index.html">
        <span class="hf-project-card__kind">Execution runtime</span>
        <h3>Holoflow</h3>
        <p>Compile declarative processing graphs into predictable CPU and GPU execution.</p>
        <span class="hf-project-card__link">Explore Holoflow <span aria-hidden="true">→</span></span>
      </a>
    </div>

    <div class="hf-projects hf-projects--secondary">
      <a class="hf-project-card hf-project-card--small" href="curaii/index.html">
        <span class="hf-project-card__kind">GPU infrastructure</span>
        <h3>Curaii</h3>
        <p>RAII wrappers for CUDA resources and companion libraries.</p>
        <span class="hf-project-card__link">Explore Curaii <span aria-hidden="true">→</span></span>
      </a>
      <a class="hf-project-card hf-project-card--small" href="holofile/index.html">
        <span class="hf-project-card__kind">Acquisition format</span>
        <h3>Holofile</h3>
        <p>High-throughput reading and writing of holographic recordings.</p>
        <span class="hf-project-card__link">Explore Holofile <span aria-hidden="true">→</span></span>
      </a>
    </div>

    <p class="hf-supporting">
      Supporting libraries: <code>holotask</code> for reusable operators,
      <code>holonp</code> for numerical primitives, and <code>holoflow_event</code>
      for runtime communication.
    </p>
  </div>
</section>

<section class="hf-section hf-section--author" aria-labelledby="authors-title">
  <div class="hf-shell">
    <header class="hf-section__heading">
      <p class="hf-eyebrow">Built by</p>
      <h2 id="authors-title">Physics and engineering, developed together</h2>
    </header>
    <div class="hf-authors">
      <article class="hf-author-card">
        <img
          class="hf-author__portrait"
          src="assets/images/michael-atlan.svg"
          alt="Michael Atlan"
          width="150"
          height="150"
          loading="lazy"
        >
        <div class="hf-author__copy">
          <h3>Michael Atlan</h3>
          <p class="hf-author__role">Scientific lead · Tenured Researcher at CNRS</p>
          <hr>
          <p>
            Michael defines the scientific direction across digital holography, Doppler imaging,
            optical coherence tomography, and ophthalmic experiments.
          </p>
          <a class="hf-text-link" href="https://www.pariseyeimaging.com/Members/180ab8642a-Michael-Atlan.en.htm">Research profile <span aria-hidden="true">→</span></a>
        </div>
      </article>
      <article class="hf-author-card">
        <img
          class="hf-author__portrait"
          src="assets/images/jules-guillou.svg"
          alt="Jules Guillou"
          width="420"
          height="420"
          loading="lazy"
        >
        <div class="hf-author__copy">
          <h3>Jules Guillou</h3>
          <p class="hf-author__role">Creator and lead developer</p>
          <hr>
          <p>
            Jules designs and implements the software stack, from the Holoflow execution model and
            CUDA processing to the Holovibes application.
          </p>
          <a class="hf-text-link" href="https://github.com/JulesGuillou">GitHub profile <span aria-hidden="true">→</span></a>
        </div>
      </article>
    </div>
  </div>
</section>

<section class="hf-closing" aria-labelledby="closing-title">
  <div class="hf-shell hf-closing__inner">
    <div>
      <p class="hf-eyebrow">Where to begin</p>
      <h2 id="closing-title">Enter through the experiment or the engine.</h2>
    </div>
    <div class="hf-actions">
      <a class="md-button md-button--primary" href="holovibes/index.html">Explore Holovibes</a>
      <a class="md-button" href="holoflow/index.html">Understand Holoflow</a>
    </div>
  </div>
</section>
