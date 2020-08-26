"""Functions to manipulate and validate configurations."""

import jsonschema
import six


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

def update_config(a, b, mode='merge'):
    """Update the configuration a with b."""
    if mode == 'merge':
        return merge_config(a, b)
    if mode == 'replace':
        return replace_config(a, b)
    raise ValueError('Invalid configuration update mode: %s' % mode)

def index_config(config, path, index_structure=True):
    """Index a configuration with a path-like string."""
    key = None
    sections = path.split('/')
    if not index_structure:
        key = sections[-1]
        sections = sections[:-1]
    for section in sections:
        if isinstance(config, dict):
            if section not in config:
                raise ValueError('Invalid path %s in config' % path)
            config = config[section]
        elif isinstance(config, list):
            try:
                section_index = int(section)
            except ValueError:
                raise ValueError('Expected an array index in path, but got %s instead' % section)
            config = config[section_index]
        else:
            raise ValueError('Paths in config can only represent object and array structures')
    if index_structure:
        return config
    else:
        return config, key

def index_schema(schema, path):
    """Index a JSON schema with a path-like string."""
    for section in path.split('/'):
        if schema['type'] != 'object':
            raise ValueError('Only object types are supported in the schema structure, '
                             'but saw type %s' % schema['type'])
        properties = schema['properties']
        if section not in properties:
            raise ValueError('Invalid path %s in user options' % path)
        schema = properties[section]
    return schema

def validate_inference_options(inference_options, config):
    """Validate the inference options, raising ValueError on error."""
    json_schema = inference_options.get('json_schema')
    if json_schema is None:
        raise ValueError('Missing "json_schema" in "inference_options"')
    jsonschema.Draft7Validator.check_schema(json_schema)
    options = inference_options.get('options')
    if options is None:
        raise ValueError('Missing "options" in "inference_options"')
    validate_mapping(json_schema, options, config)
    return json_schema

def validate_mapping(schema, options, config):
    """Validate the mapping between inference options and configuration fields,
    raising ValueError on error.
    """
    for i, mapping in enumerate(options):
        config_path = mapping.get('config_path')
        if config_path is None:
            raise ValueError('Missing "config_path" in option mapping %d' % i)
        dst_config, _ = index_config(config, config_path, index_structure=False)
        if not isinstance(dst_config, dict):
            raise ValueError('Paths in config can only index object structures')
        option_path = mapping.get('option_path')
        if option_path is None:
            raise ValueError('Missing "option_path" in option mapping %d' % i)
        option_schema = index_schema(schema, option_path)

def update_config_with_options(config, options):
    """Update the configuration with incoming inference options. Raises ValueError
    if inference options were not expected or the value is not accepted.
    """
    inference_options = config.get('inference_options')
    if inference_options is None:
        raise ValueError('This model does not expect inference options')
    try:
        jsonschema.validate(options, inference_options['json_schema'])
    except jsonschema.ValidationError as e:
        raise ValueError('Options validation error: %s' % e.message)
    for mapping in inference_options['options']:
        try:
            option_value = index_config(options, mapping['option_path'])
        except ValueError:
            continue  # Option not passed for this request.
        dst_config, dst_key = index_config(config, mapping['config_path'], index_structure=False)
        dst_config[dst_key] = option_value


def old_to_new_config(config):
    # old configurations
    if not config:
        return
    tok_config = config.get("tokenization")
    if tok_config:
        vocab_src = tok_config["source"].get("vocabulary", None)
        vocab_tgt = tok_config["target"].get("vocabulary", None)
        replace_src = tok_config["source"].get("replace_vocab", False)
        replace_tgt = tok_config["target"].get("replace_vocab", False)
        if vocab_src or vocab_tgt:
            if "vocabulary" not in config:
                config["vocabulary"] = {}
            if vocab_src:
                if "source" not in config["vocabulary"]:
                    config["vocabulary"]["source"] = { "path": vocab_src }
                else:
                    config["vocabulary"]["source"]["path"] = vocab_src
                config["vocabulary"]["source"]["replace_vocab"] = replace_src
            if vocab_tgt:
                if "target" not in config["vocabulary"]:
                    config["vocabulary"]["target"] = { "path": vocab_tgt }
                else:
                    config["vocabulary"]["target"]["path"] = vocab_tgt
                config["vocabulary"]["target"]["replace_vocab"] = replace_tgt


        config["preprocess"] = [
            {
                "op":"tokenization",
                "source": tok_config["source"],
                "target": tok_config["target"]
            }
        ]
