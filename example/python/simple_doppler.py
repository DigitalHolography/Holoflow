"""PCA Doppler processing using a traced native streaming graph.

Usage (with the CMake-built holoflow package on PYTHONPATH):
    python simple_doppler.py input.holo output.holo

Optical parameters, queue capacities, and component bounds are illustrative.
The input must contain at least one complete 256-frame temporal window.
The file source loops; recording completion is followed by an explicit stop.
"""

import argparse
import time

import holoflow as hf


def simple_doppler(
    input_path,
    output_path,
    *,
    time_window=256,
    component_begin=4,
    component_end=64,
    accumulation=4,
):
    frames = hf.sources.holofile(
        input_path,
        batch_size=32,
        load_kind="live",
    )

    frames = frames.to("device")
    frames = hf.batch_queue(
        frames,
        target_capacity=1024,
        output_size=time_window,
        output_stride=time_window,
    )
    frames = frames.astype(hf.float32)  # [T, H, W]

    # Batch four temporal windows together, as in the C++ pipeline.
    t, h, w = frames.shape
    frames = frames.reshape((1, t, h, w))
    frames = hf.batch_queue(
        frames,
        target_capacity=8,
        output_size=4,
        output_stride=4,
    )  # [4, T, H, W]

    components = hf.pca(
        frames,
        begin=component_begin,
        end=component_end,
    )  # [4, component_end - component_begin, H, W]

    magnitude = hf.fresnel_diffraction(
        components,
        wavelength=532e-9,
        dx=12e-6,
        dy=12e-6,
        z=0.20,
        axes=(-2, -1),
        output_magnitude=True,
    )

    # Match the C++ M0 path: mean magnitude, then mean across windows.
    image = hf.mean(magnitude, axis=-3, keepdims=True)
    image = hf.mean(image, axis=0)  # [1, H, W]
    image = hf.sliding_average(
        image,
        target_capacity=8,
        window_size=accumulation,
        discard_first=0,
    )

    image = image.astype(hf.uint8, strategy="scaled").to("host")
    image = hf.batch_queue(
        image,
        target_capacity=8,
        output_size=1,
        output_stride=1,
    )
    hf.sinks.holofile(
        image,
        output_path,
        count=100,
        node_id="doppler_writer",
    )


def main():
    parser = argparse.ArgumentParser(description="PCA Doppler processing")
    parser.add_argument("input", help="Input .holo file")
    parser.add_argument("output", help="Output .holo file")
    args = parser.parse_args()

    graph = hf.trace(
        simple_doppler,
        input_path=args.input,
        output_path=args.output,
    )

    with graph.compile() as executable:
        executable.start()
        # Events use the native sink protocol and its explicit graph node ID.
        # Acceptance into the outgoing queue does not acknowledge handling.
        if not executable.send_event(
            "doppler_writer",
            {
                "type": "start_recording",
                "record_path": args.output,
            },
        ):
            raise RuntimeError("Could not enqueue recording command")

        deadline = time.monotonic() + 60
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Recording did not finish")

            event = executable.receive_event()
            if event is None:
                time.sleep(0.01)
                continue

            if event.node_id != "doppler_writer":
                continue

            if event.data["type"] == "recording_failed":
                raise RuntimeError(event.data["message"])
            if event.data["type"] == "recording_finished":
                break

        executable.stop()
        executable.wait()


if __name__ == "__main__":
    main()
