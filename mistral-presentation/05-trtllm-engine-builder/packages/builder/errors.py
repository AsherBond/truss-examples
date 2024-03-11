class CheckpointConversionError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"Checkpoint conversion failed with: `{self.message}`"


class EngineBuildError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"Engine build failed with: `{self.message}`"
