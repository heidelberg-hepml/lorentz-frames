PARTICLE_TYPE = {
    "aag": [0, 0, 1, 1, 0],
    "aagg": [0, 0, 1, 1, 0, 0],
    "zg": [0, 0, 1, 2],
    "zgg": [0, 0, 1, 2, 2],
    "zggg": [0, 0, 1, 2, 2, 2],
    "zgggg": [0, 0, 1, 2, 2, 2, 2],
    "zggggg": [0, 0, 1, 2, 2, 2, 2, 2],
}
DATASET_TITLE = {
    "aag": r"$gg\to\gamma\gamma g$",
    "aagg": r"$gg\to\gamma\gamma gg$",
    "zg": r"$q\bar q\to Zg$",
    "zgg": r"$q\bar q\to Zgg$",
    "zggg": r"$q\bar q\to Zggg$",
    "zgggg": r"$q\bar q\to Zgggg$",
    "zggggg": r"$q\bar q \to Zggggg$",
}

mass_Z = 91.188


def get_mass(dataset, mass_reg):
    assert dataset in PARTICLE_TYPE.keys()

    # initialize massless particles
    mass = [mass_reg] * (len(dataset) + 2)

    # if included, z comes in 3rd position
    if "z" in dataset:
        mass[2] = mass_Z
    return mass
