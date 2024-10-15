#! /usr/bin/env python
import os

MODELS = ["base_car", "pusher_car"]

MAPPING = ["{C}", "{InitX}", "{InitY}", "{InitT}"]
ENTRIES = [
    ("goose", 0.0, 3.0, 0.0),
    ("buddy", 0.0, 2.0, 0.0),
    ("car1", 0.0, 1.0, 0.0),
    ("car2", 0.0, 0.0, 0.0),
    ("car3", 0.0, -1.0, 0.0),
    ("car4", 0.0, -2.0, 0.0),
    ("car5", 0.0, -3.0, 0.0),
]

if __name__ == "__main__":
    for model in MODELS:
        with open(model + ".template", "rb") as f:
            car_data = f.readlines()

        for e in ENTRIES:
            if not os.path.exists(model):
                os.mkdir(model)

            name = e[MAPPING.index("{C}")]
            print("Formatting {}/{}.xml".format(model, name))
            with open("{}/{}.xml".format(model, name), "wb") as f:
                for line in car_data:
                    for k, v in zip(MAPPING, e):
                        line = line.replace(k, str(v))
                    f.write(line)
