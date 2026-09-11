"""View a holotask NPY recording: python tools/view_npy.py recording.npy"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def main() -> None:
    parser = argparse.ArgumentParser(description="View a (frames, height, width) NPY recording")
    parser.add_argument("path")
    args = parser.parse_args()

    try:
        data = np.load(args.path, mmap_mode="r")
    except Exception as error:
        parser.error(f"could not load '{args.path}': {error}")
    if data.ndim != 3 or data.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        parser.error(f"expected a uint8/uint16 rank-3 array, got shape={data.shape}, dtype={data.dtype}")

    frames, height, width = data.shape
    print(f"shape: {data.shape}")
    print(f"dtype: {data.dtype}")
    print(f"frames: {frames}")
    print(f"frame dimensions: {width} x {height}")

    image = plt.imshow(data[0], cmap="gray", vmin=0, vmax=np.iinfo(data.dtype).max)
    title = plt.title(f"Frame 1/{frames}")
    plt.axis("off")
    current = {"index": 0}
    playing = {"value": True}
    animation = None

    def show(index: int):
        current["index"] = index % frames
        image.set_data(data[current["index"]])
        title.set_text(f"Frame {current['index'] + 1}/{frames}")
        return image, title

    def key(event):
        if event.key == "space":
            if playing["value"]:
                animation.pause()
            else:
                animation.resume()
            playing["value"] = not playing["value"]
        elif event.key in ("right", "n"):
            show(current["index"] + 1)
        elif event.key in ("left", "p"):
            show(current["index"] - 1)
        elif event.key == "home":
            show(0)
        fig.canvas.draw_idle()

    fig = plt.gcf()
    fig.canvas.mpl_connect("key_press_event", key)
    animation = FuncAnimation(fig, lambda frame: show(frame), frames=frames, interval=100, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
