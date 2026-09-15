# Optical Heterodyne Detection

Optical heterodyne detection mixes a weak signal field with a frequency-shifted local oscillator (LO). Their optical carriers are much too fast for a camera to follow, but their interference contains a difference-frequency term that can fall inside the camera bandwidth.

This lesson first treats the fields themselves, then introduces the square-law detector that turns their interference into a measurable intensity.

## Two fields and their beat

Consider two real, co-polarized fields at one point on the detector:

\[
s_\mathrm{sig}(t)=A_\mathrm{sig}\cos(2\pi f_\mathrm{sig}t+\phi_\mathrm{sig}),
\qquad
s_\mathrm{LO}(t)=A_\mathrm{LO}\cos(2\pi f_\mathrm{LO}t+\phi_\mathrm{LO}).
\]

In the animations, `s_beat` denotes their summed optical field:

\[
s_\mathrm{beat}(t)=s_\mathrm{sig}(t)+s_\mathrm{LO}(t),
\qquad
\Delta f=f_\mathrm{LO}-f_\mathrm{sig}.
\]

Using the sum-to-product identity

\[
\cos a+\cos b
=
2\cos\!\left(\frac{a-b}{2}\right)
\cos\!\left(\frac{a+b}{2}\right),
\]

the equal-amplitude, equal-phase sum can be factored as

\[
s_\mathrm{beat}(t)=2A
\cos\!\left(\pi\Delta f\,t\right)
\cos\!\left(2\pi\frac{f_\mathrm{sig}+f_\mathrm{LO}}{2}t\right).
\]

The rapidly oscillating carrier therefore sits inside a slowly changing envelope.

<figure>
  <video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: auto;" aria-label="Two equal-amplitude fields with a fixed 20 kilohertz detuning and their summed beat field">
    <source src="../assets/videos/learn/heterodyne/heterodyne-fixed-detuning.mp4" type="video/mp4">
  </video>
  <figcaption>Equal field amplitudes with a fixed 20 kHz detuning. The carrier frequencies are compressed for display.</figcaption>
</figure>

## Detuning sets the beat frequency

Changing \(\Delta f\) changes the rate at which the two fields move into and out of phase. Small detuning produces a slow envelope; larger detuning produces more envelope cycles in the same observation time.

The next animation holds both amplitudes fixed while sweeping the displayed LO detuning between 5 and 35 kHz. The phase evolves continuously throughout the loop.

<figure>
  <video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: auto;" aria-label="Two equal-amplitude fields with variable frequency detuning and their summed beat field">
    <source src="../assets/videos/learn/heterodyne/heterodyne-variable-detuning.mp4" type="video/mp4">
  </video>
  <figcaption>Increasing the detuning makes the beat envelope evolve more rapidly.</figcaption>
</figure>

## What the local oscillator amplifies

Increasing \(A_\mathrm{LO}\) increases the amplitude of the combined field, but it does not amplify the original signal field. Heterodyne gain appears when a square-law detector measures the irradiance

\[
I_\mathrm{raw}(t)\propto
\left[s_\mathrm{sig}(t)+s_\mathrm{LO}(t)\right]^2.
\]

A detector that averages over many optical periods removes terms near twice the optical carrier. The remaining signal is

\[
\overline I(t)=
\frac{A_\mathrm{sig}^{2}+A_\mathrm{LO}^{2}}{2}
+A_\mathrm{sig}A_\mathrm{LO}
\cos\!\left(2\pi\Delta f\,t+\Delta\phi\right),
\]

where \(\Delta\phi=\phi_\mathrm{LO}-\phi_\mathrm{sig}\). The useful heterodyne cross-term has amplitude \(A_\mathrm{sig}A_\mathrm{LO}\): for a fixed weak signal, a stronger LO produces a larger absolute modulation at the detector. The LO also contributes a DC term proportional to \(A_\mathrm{LO}^2\), so more LO power is not an unlimited improvement—detector saturation, shot noise, and technical noise still matter.

<figure>
  <video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: auto;" aria-label="Fixed signal field mixed with a variable-amplitude local oscillator and their summed beat field">
    <source src="../assets/videos/learn/heterodyne/heterodyne-lo-amplitude.mp4" type="video/mp4">
  </video>
  <figcaption>The LO field amplitude varies while the signal and 20 kHz detuning remain fixed. The annotation tracks the square-law cross-term amplitude.</figcaption>
</figure>

## From an optical carrier to camera samples

At 852 nm, the optical carrier frequency is approximately

\[
f_0=\frac{c}{\lambda}\approx 351.87\ \mathrm{THz}.
\]

No conventional camera samples that field oscillation directly. A pixel integrates irradiance during an exposure of duration \(T_\mathrm{exp}\). For a rectangular exposure centered on \(t\), the beat term is attenuated by the exposure transfer function:

\[
\overline I_{T_\mathrm{exp}}(t)=
\frac{A_\mathrm{sig}^{2}+A_\mathrm{LO}^{2}}{2}
+A_\mathrm{sig}A_\mathrm{LO}
\operatorname{sinc}(\Delta f T_\mathrm{exp})
\cos\!\left(2\pi\Delta f\,t+\Delta\phi\right),
\]

with \(\operatorname{sinc}(x)=\sin(\pi x)/(\pi x)\). The animation uses \(\Delta f=20\ \mathrm{kHz}\), a 50 kfps camera, and a full-frame \(20\ \mu\mathrm{s}\) exposure. The beat attenuation is therefore \(\operatorname{sinc}(0.4)\approx0.757\). Its 25 kHz Nyquist frequency remains above the 20 kHz beat.

The displayed 2 MHz carrier is still an enormous compression of the physical 351.87 THz carrier. It exists only to make the raw oscillations drawable; the averaged curve is calculated from the analytic exposure model above.

<figure>
  <video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: auto;" aria-label="Optical fields, raw square-law intensity, and exposure-averaged samples from a 50 kiloframe-per-second camera">
    <source src="../assets/videos/learn/heterodyne/heterodyne-camera-averaging.mp4" type="video/mp4">
  </video>
  <figcaption>Finite exposure suppresses the fast carrier terms and leaves a clean, sampled difference-frequency modulation.</figcaption>
</figure>

!!! note "What if the camera runs at 37 kfps?"
    A 37 kfps camera has a Nyquist frequency of 18.5 kHz. A 20 kHz beat is above that limit and aliases to \(|20-37|=17\ \mathrm{kHz}\). Exposure averaging still rejects the optical carrier, but an unambiguous 20 kHz measurement requires a higher frame rate or a lower detuning.

## Takeaways

- The sum of two nearby-frequency fields has a beat envelope governed by \(|\Delta f|\).
- A square-law detector creates a low-frequency cross-term proportional to \(A_\mathrm{sig}A_\mathrm{LO}\).
- Exposure integration rejects optical-frequency oscillations but attenuates the beat according to a sinc response.
- The camera frame rate must still satisfy the sampling requirements of the difference-frequency signal.
