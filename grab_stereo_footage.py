from __future__ import annotations

import numpy as np

from stereo_pair import StereoPair, CameraType


LEFT_SERIAL  = "09H3Z"
LEFT_TYPE    = CameraType.ALLIED_VISION

RIGHT_SERIAL = "24856864"
RIGHT_TYPE   = CameraType.BASLER

NUM_FRAMES = 10


def main():

    drifts_ms = []

    with StereoPair(
        serial_left=LEFT_SERIAL,
        type_left=LEFT_TYPE,

        serial_right=RIGHT_SERIAL,
        type_right=RIGHT_TYPE,

        exposure_us=350.0,
        gain_db=10.0,

        use_external_trigger=True,

        trigger_source_left="Line1",
        trigger_source_right="Line1",
    ) as pair:

        print("Measuring synchronization drift...")
        print()

        for i in range(NUM_FRAMES):

            (
                left_img,
                right_img,
                left_ts,
                right_ts,
            ) = pair.grab()

            if left_img is None or right_img is None:
                print("Grab failed")
                continue

            # Convert ns → ms
            drift_ms = (left_ts - right_ts) / 1e6

            drifts_ms.append(drift_ms)

            print(
                f"[{i:03d}] "
                f"Left={left_ts} ns   "
                f"Right={right_ts} ns   "
                f"Drift={drift_ms:.6f} ms"
            )

    print()
    print("────────────────────────────")

    print(f"Mean drift : {np.mean(drifts_ms):.6f} ms")
    print(f"Std drift  : {np.std(drifts_ms):.6f} ms")
    print(f"Max drift  : {np.max(np.abs(drifts_ms)):.6f} ms")

    print("────────────────────────────")


if __name__ == "__main__":
    main()