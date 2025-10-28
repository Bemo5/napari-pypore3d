from importlib.resources import files

# expose manifest path for npe2
def __getattr__(name: str):
    if name == "napari.yaml":
        return str(files(__package__) / "napari.yaml")
    raise AttributeError(name)
