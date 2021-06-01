import copy

from nmtwizard.preprocess import prepoperator


@prepoperator.register_operator("identity_filter")
class IdentityFilter(prepoperator.Filter):
    """Ignore TU with the same source and target."""

    @classmethod
    def _config_schema(cls):
        schema = super(IdentityFilter, cls)._config_schema()

        schema["properties"].update(
            {"min_characters": {"type": "integer", "minimum": 0}}
        )
        return schema

    def __init__(self, config, *args, **kwargs):
        # Do not ignore identity TU if it has less than this number of characters.
        min_characters = config.get("min_characters", 0)

        filter_fn = (
            lambda tu: len(tu.src_detok) > min_characters
            and tu.src_detok == tu.tgt_detok
        )
        super().__init__([filter_fn])
