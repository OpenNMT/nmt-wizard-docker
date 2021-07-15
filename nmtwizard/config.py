"""Functions to manipulate and validate configurations."""

import collections
import jsonschema
import six
import copy


def merge_config(a, b):
    """Merges config b in a."""
    for key, b_value in six.iteritems(b):
        if not isinstance(b_value, dict):
            a[key] = b_value
        else:
            a_value = a.get(key)
            if a_value is not None and isinstance(a_value, dict):
                merge_config(a_value, b_value)
            else:
                a[key] = b_value
    return a


def replace_config(a, b):
    """Updates fields in a by fields in b."""
    a.update(b)
    return a


_non_user_fields = {"model", "modelType", "imageTag", "build", "parent_model"}


def update_config(a, b, mode="default"):
    """Update the configuration a with b."""
    if not b:
        return a

    from_version = get_config_version(a)
    to_version = get_config_version(b)
    if from_version == 1 and to_version == 2:
        # When updating the configuration to a newer version, we clear all user fields.
        a = {k: v for k, v in a.items() if k in _non_user_fields}
        return replace_config(a, b)

    if mode == "default" or mode == "merge":
        return merge_config(a, b)
    if mode == "replace":
        return replace_config(a, b)
    raise ValueError("Invalid configuration update mode: %s" % mode)


def index_config(config, path, index_structure=True):
    """Index a configuration with a path-like string."""
    key = None
    sections = path.split("/")
    if not index_structure:
        key = sections[-1]
        sections = sections[:-1]
    for section in sections:
        if isinstance(config, dict):
            if section not in config:
                raise ValueError("Invalid path %s in config" % path)
            config = config[section]
        elif isinstance(config, list):
            section_index = None
            try:
                section_index = int(section)
            except ValueError:
                for i, block in enumerate(config):
                    if isinstance(block, dict) and block.get("name") == section:
                        section_index = i
                        break
                if section_index is None:
                    raise ValueError(
                        "Expected an array index in path, but got %s instead" % section
                    )
            config = config[section_index]
        else:
            raise ValueError(
                "Paths in config can only represent object and array structures"
            )
    if index_structure:
        return config
    else:
        return config, key


def build_override(config, path, value):
    """Builds a configuration override to update the value at path."""
    if not path:
        return value
    sections = path.split("/")
    section = sections[0]
    inner_path = "/".join(sections[1:])
    if isinstance(config, dict):
        return {section: build_override(config.get(section), inner_path, value)}
    if isinstance(config, list):
        index = int(sections[0])
        override = build_override(config[index], inner_path, value)
        # Since lists can't be merged, the override should contain the full list content.
        config = list(config)
        if isinstance(override, dict):
            config[index] = merge_config(copy.deepcopy(config[index]), override)
        else:
            config[index] = override
        return config
    raise TypeError("Paths in config can only represent object and array structures")


def index_schema(schema, path):
    """Index a JSON schema with a path-like string."""
    for section in path.split("/"):
        if schema["type"] != "object":
            raise ValueError(
                "Only object types are supported in the schema structure, "
                "but saw type %s" % schema["type"]
            )
        properties = schema["properties"]
        if section not in properties:
            raise ValueError("Invalid path %s in user options" % path)
        schema = properties[section]
    return schema


def validate_inference_options(inference_options, config):
    """Validate the inference options, raising ValueError on error."""
    json_schema = inference_options.get("json_schema")
    if json_schema is None:
        raise ValueError('Missing "json_schema" in "inference_options"')
    jsonschema.Draft7Validator.check_schema(json_schema)
    options = inference_options.get("options")
    if options is None:
        raise ValueError('Missing "options" in "inference_options"')
    validate_mapping(json_schema, options, config)
    return json_schema


def validate_mapping(schema, options, config):
    """Validate the mapping between inference options and configuration fields,
    raising ValueError on error.
    """
    for i, mapping in enumerate(options):
        config_path = mapping.get("config_path")
        if config_path is None:
            raise ValueError('Missing "config_path" in option mapping %d' % i)
        if isinstance(config_path, str):
            config_path = [config_path]
        for cp in config_path:
            dst_config, _ = index_config(config, cp, index_structure=False)
            if not isinstance(dst_config, dict):
                raise ValueError("Paths in config can only index object structures")
        option_path = mapping.get("option_path")
        if option_path is None:
            raise ValueError('Missing "option_path" in option mapping %d' % i)
        _ = index_schema(schema, option_path)


def read_options(config, options):
    """Reads the inference options.

    For V1 configurations, this function returns a configuration override.
    For V2 configurations, this function returns a dict mapping operator names to their options.

    Raises:
      ValueError: if inference options were not expected or the value is not accepted.
    """
    inference_options = config.get("inference_options")
    if inference_options is None:
        raise ValueError("This model does not expect inference options")
    try:
        jsonschema.validate(options, inference_options["json_schema"])
    except jsonschema.ValidationError as e:
        raise ValueError("Options validation error: %s" % e.message)
    v2_config = is_v2_config(config)
    operators_options = collections.defaultdict(dict)
    config_override = {}
    for mapping in inference_options["options"]:
        try:
            option_value = index_config(options, mapping["option_path"])
        except ValueError:
            continue  # Option not passed for this request.
        config_path = mapping["config_path"]
        if isinstance(config_path, str):
            config_path = [config_path]
        if v2_config:
            for cp in config_path:
                dst_config, dst_key = index_config(config, cp, index_structure=False)
                operators_options[dst_config["name"]].update({dst_key: option_value})
        else:
            for cp in config_path:
                merge_config(
                    config_override,
                    build_override(config, cp, option_value),
                )
    if v2_config:
        return operators_options
    return config_override


def is_v2_config(config):
    """Returns True if config is a V2 configuration."""
    preprocess = config.get("preprocess")
    return (
        "tokenization" not in config
        and preprocess is not None
        and isinstance(preprocess, list)
    )


def is_v1_config(config):
    """Returns True if config is a V1 configuration."""
    return not is_v2_config(config)


def get_config_version(config):
    """Returns the version of the configuration."""
    return 2 if is_v2_config(config) else 1


def ensure_operators_name(config):
    """Make sure all operators in model configuration have a unique name."""
    if is_v1_config(config):
        return
    i = 1
    for process in ["preprocess", "postprocess"]:
        process_config = config.get(process)
        if process_config:
            for op_config in process_config:
                op_type = op_config.get("op")
                if op_type:
                    op_config.setdefault("name", "%s_%d" % (op_type, i))
                i += 1


def old_to_new_config(config):
    """Locally update old configuration with 'tokenization' field to include new 'vocabulary' and 'preprocess" fields."""
    if not config:
        return
    tok_config = config.get("tokenization")
    new_config = config
    if tok_config:
        if "vocabulary" not in config:
            new_config = copy.deepcopy(config)
            vocab_src = tok_config["source"].get("vocabulary", None)
            vocab_tgt = tok_config["target"].get("vocabulary", None)
            replace_src = tok_config["source"].get("replace_vocab", False)
            replace_tgt = tok_config["target"].get("replace_vocab", False)
            prev_vocab_src = tok_config["source"].get("previous_vocabulary", None)
            prev_vocab_tgt = tok_config["target"].get("previous_vocabulary", None)

            if vocab_src or vocab_tgt:
                new_config["vocabulary"] = {}
            if vocab_src:
                new_config["vocabulary"]["source"] = {
                    "path": vocab_src,
                    "replace_vocab": replace_src,
                }
            if vocab_tgt:
                new_config["vocabulary"]["target"] = {
                    "path": vocab_tgt,
                    "replace_vocab": replace_tgt,
                }
            if prev_vocab_src:
                new_config["vocabulary"]["source"][
                    "previous_vocabulary"
                ] = prev_vocab_src
            if prev_vocab_tgt:
                new_config["vocabulary"]["target"][
                    "previous_vocabulary"
                ] = prev_vocab_tgt

        if "preprocess" not in config:
            new_tok_config = copy.deepcopy(tok_config)
            new_tok_config["source"].pop("vocabulary", None)
            new_tok_config["target"].pop("vocabulary", None)
            new_tok_config["source"].pop("replace_vocab", None)
            new_tok_config["target"].pop("replace_vocab", None)
            new_config["preprocess"] = [
                {
                    "op": "tokenization",
                    "source": new_tok_config["source"],
                    "target": new_tok_config["target"],
                }
            ]

    return new_config


def _ensure_params_order(params):
    params = collections.OrderedDict(sorted(params.items(), key=lambda x: x[0]))
    preferred_first = ["op", "name"]
    preferred_last = ["overrides"]

    for field in reversed(preferred_first):
        if field in params:
            params.move_to_end(field, last=False)

    for field in preferred_last:
        if field in params:
            params.move_to_end(field, last=True)

    return params


def prepare_config_for_save(config):
    """Prepares the configuration before saving it in the model directory."""
    if is_v2_config(config):
        # In V2 operators, we prefer that some fields appear first (or last) for readability.
        config = config.copy()
        for section_name in ("preprocess", "postprocess"):
            section = config.get(section_name)
            if section is None:
                continue
            config[section_name] = [_ensure_params_order(params) for params in section]

    return config
