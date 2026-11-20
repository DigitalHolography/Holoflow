# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Build the demo graph independently of YAML and terminal reporting."""

import holoflow as hf


def _flatfield(image, cutoff, wavelength, pitch, distance):
    height, width = image.shape[-2:]
    scale = 0.187 * cutoff * pitch / (wavelength * abs(distance))
    return hf.flatfield(image, scale * height, scale * width)


def _host_queue(image, config, node_id):
    return hf.batch_queue(image.to("host"), config.queues["output_images"], 1, 1,
                          node_id=node_id)


def _correct_aberration(field, config, wavelength, pitch):
    settings = config.aberration
    window = config.correction_window
    height, width = field.shape[-2:]
    win_h = height // settings["subapertures"]
    win_w = width // settings["subapertures"]
    if min(win_h, win_w) < 2:
        raise ValueError("aberration.subapertures must leave at least two pixels per subaperture")
    distance = settings["distance_m"]
    if settings["prefilter"] is not None:
        field = hf.filter2d(field, **settings["prefilter"])
    current, delayed, _ = hf.dual_reader_batch_queue(
        field, config.queues["propagation_windows"], window, node_id="correction_input")
    sensor = hf.short_time_fresnel_diffraction(
        current, wavelength, pitch, pitch, distance, win_h, win_w, win_h, win_w)
    sensor = hf.fftshift(hf.causal_sliding_average(hf.mean(sensor, axis=1), window))
    if (cutoff := settings["sensor_flatfield_cutoff_m"]) is not None:
        sensor = _flatfield(sensor, cutoff, wavelength, pitch, distance)
    skip = settings["skip_outside_pupil"]
    slopes = hf.shack_hartmann_slopes(
        sensor, wavelength, pitch, pitch, distance, win_h, win_w, win_h, win_w,
        skip_subapertures_outside_pupil=skip)
    modes = settings["zernike_modes"]
    coefficients = hf.zernike_from_slopes(
        slopes, modes, wavelength, pitch, pitch, win_h, win_w, win_h, win_w,
        skip_subapertures_outside_pupil=skip)[0, 0, :]
    phase = hf.zernike_phase(coefficients, modes, height, width)
    corrected = hf.correct_phase(delayed, phase)
    corrected = hf.batch_queue(corrected, config.queues["corrected_windows"], 1, 1,
                               node_id="corrected_windows")
    diagnostics = {}
    if config.viewer["cao_diagnostics"]:
        count = config.temporal_count
        mosaic = hf.normalize(sensor, axes=(-2, -1), lo=0, hi=255)
        _, rows, columns, _, _ = mosaic.shape
        mosaic = hf.transpose(mosaic, (0, 1, 3, 2, 4))
        mosaic = mosaic.reshape((1, rows * win_h, columns * win_w)).astype(hf.uint8, "scaled")
        wrapped = hf.wrap2pi(hf.copy(phase)).reshape((1, height, width))
        fields = {"shack_hartmann": mosaic, "phase": wrapped,
                  "coefficients": coefficients.reshape((1, len(modes)))}
        for name, value in fields.items():
            # Keep host transfers out of the CAO CUDA graph section.
            queued = hf.batch_queue(value, config.queues["output_images"], 1, 1,
                                    node_id=name + "_queue_device")
            host = _host_queue(queued, config, name + "_queue")
            diagnostics[name] = hf.sinks.numpy(host, length=count)
    return corrected, diagnostics


def pipeline(config):
    source = config.input
    interval = config.metrics["interval_ms"]
    temporal_frames = config.temporal_frames
    frames = hf.sources.holofile(
        source["path"], batch_size=source["batch_size"], load_kind=source["load_kind"],
        start_frame=source["start_frame"], end_frame=source["end_frame"],
        max_fps=source["max_fps"], loop=source["mode"] == "looping", tail=source["tail"],
    )
    frames = hf.latency_start(frames, node_id="ingress",
                              history_batches=config.metrics["history_batches"])
    frames = hf.throughput(frames, interval_ms=interval, node_id="input_fps")
    # A cached GPU source owns its output, while BatchQueue owns its input.
    frames = hf.copy(frames) if frames.device == "device" else frames.to("device")
    frames = hf.batch_queue(frames, config.queues["input_frames"], temporal_frames, temporal_frames,
                            node_id="uploaded_frames")
    frames = frames.astype(hf.float32)
    height, width = frames.shape[-2:]
    field = frames.reshape((1, temporal_frames, height, width), copy=False)
    if config.time["method"] == "pca":
        batch = config.time["batch_windows"]
        field = hf.batch_queue(field, config.queues["preprocessing_windows"], batch, batch,
                               node_id="pca_windows")
        field = hf.pca(field, config.time["component_begin"], config.time["component_end"])
        if not config.aberration["enabled"]:
            field = hf.batch_queue(field, config.queues["propagation_windows"], 1, 1,
                                   node_id="propagation_windows")

    wavelength = config.optics["wavelength_m"]
    pitch = config.optics["pixel_pitch_m"]
    diagnostics = {}
    if config.aberration["enabled"]:
        field, diagnostics = _correct_aberration(field, config, wavelength, pitch)
    if config.space["method"] == "fresnel":
        field = hf.fresnel_diffraction(
            field, wavelength, pitch, pitch, config.space["distance_m"], output_magnitude=True)
    elif config.aberration["enabled"] or config.time["method"] == "pca":
        field = hf.abs(field)
    image = hf.mean(field, axis=1)
    if config.space["method"] == "fresnel":
        image = hf.fftshift(image)
        if (cutoff := config.output["flatfield_cutoff_m"]) is not None:
            image = _flatfield(image, cutoff, wavelength, pitch, config.space["distance_m"])
    average_window = config.output["average_window"]
    if config.correction_window > 1 or average_window > 1:
        image = hf.sliding_average(image, 8, average_window,
                                   discard_first=config.correction_window - 1)
    if (clip := config.output["percentile_clip"]) is not None:
        image = hf.pct_clip(image, **clip)
    if config.output["dtype"] == "uint8":
        image = image.astype(hf.uint8, "scaled")
    image = _host_queue(image, config, "host_output")
    image = hf.throughput(image, interval_ms=interval, node_id="output_fps")
    measured = hf.latency_end(
        image, source_node_id="ingress", node_id="latency", interval_ms=interval,
        window_frames=(config.correction_window + average_window - 1) * temporal_frames,
        stride_frames=temporal_frames,
    )
    if config.viewer["enabled"]:
        xy = hf.sinks.numpy(measured, length=config.output_count, node_id="xy_output")
        return {"xy": xy, **diagnostics} if diagnostics else xy
    hf.sinks.void(measured, node_id="xy_output")
